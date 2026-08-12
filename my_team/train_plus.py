#!/usr/bin/env python3
"""Training driver with an enriched opponent pool.

WHY THIS EXISTS
---------------
`scripts/train.py` trains against {Random, TitForTat} + self + snapshots.
Measured behaviour of the resulting policies (analysis/probe.py):

    as U.S.: exits in 100% of episodes, mean t = 3   -- pure concession
    as Iran: escalates to sigma = 0.75 against a *mirroring* opponent
             and then exits at t ~ 4 with a large commitment stock

Both are artefacts of the pool. Random exits on its own about 1/6 of the
time, and TitForTat concedes once its endurance drops below 0.20, so the
learner is never punished for folding early -- there is no opponent in the
pool that simply refuses to leave. analysis/sweep.py confirms the cost of
that blind spot: several one-line scripted strategies beat TitForTat by
+8 to +17 points, while the trained pair loses to it by ~50.

So we inject the archetypes of `my_team/opponents.py` into the trainer's
pool. `see/training/selfplay.py` reads the pool through the module-level
name `BASELINES`, so rebinding that dictionary *in memory* is enough. No
file under `see/` is touched: the physics, the network architecture, the
PPO code and the tournament all stay canonical. Only which opponents the
learner meets during training changes -- which is precisely the knob
docs/DESIGN.md Sec. 11 says is meant to be tuned.

    python my_team/train_plus.py --theory my_team/theory.py \
        --team mine --steps 600000 --out runs/plus.pt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import see.agents.scripted as scripted                       # noqa: E402
import see.training.selfplay as selfplay                     # noqa: E402
from see.training.theory_api import load_theory              # noqa: E402
from my_team.opponents import EXTRA_BASELINES                # noqa: E402
from my_team.reference_pool import (BOARD_BASELINES,           # noqa: E402
                                    BOARD_PATIENT_BASELINES,
                                    BOARD_PLUS_BASELINES,
                                    OFFICIAL_BASELINES,
                                    OFFICIAL_PLUS_BASELINES)


def install_pool(mode: str) -> dict:
    """Rebind the trainer's opponent dictionary (in memory only)."""
    pool = dict(scripted.BASELINES)                # Random, TitForTat
    if mode == "rich":
        pool.update(EXTRA_BASELINES)
    elif mode == "board":
        # The leaderboard scores against five scripted baselines plus the
        # instructor benchmark, not against the two that ship in the repo.
        # my_team/reference_pool.py reconstructs the missing three and fits
        # them to the published rows; training against that pool is the
        # whole point of this mode.
        pool = dict(BOARD_BASELINES)
    elif mode == "board+":
        # `board` was a regression as well as an improvement: it replaced
        # the pool instead of extending it, so the patient archetypes of
        # my_team/opponents.py fell out and the learner stopped meeting any
        # opponent that simply waits. The board's own numbers show the cost
        # -- see the note above BOARD_PLUS_BASELINES.
        pool = dict(BOARD_PLUS_BASELINES)
    elif mode == "board+p":
        # board+ overshot; this is the rebalanced version. See the note
        # above BOARD_PATIENT_BASELINES.
        pool = dict(BOARD_PATIENT_BASELINES)
    elif mode == "official":
        # exactly the graded tournament's scripted opponents; the rest of
        # that field is learned agents, which self-play stands in for.
        pool = dict(OFFICIAL_BASELINES)
    elif mode == "official+":
        # the graded scripted pool plus a never-conceder, which stands in
        # for the classmates who will submit one
        pool = dict(OFFICIAL_PLUS_BASELINES)
    scripted.BASELINES = pool
    selfplay.BASELINES = pool                     # already imported by name
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", required=True)
    ap.add_argument("--team", default="anonymous")
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/plus.pt")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.025)
    ap.add_argument("--p-self", type=float, default=0.30)
    ap.add_argument("--p-scripted", type=float, default=0.80)
    ap.add_argument("--pool",
                    choices=("stock", "rich", "board", "board+",
                             "board+p", "official", "official+"),
                    default="rich")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save-every", type=int, default=150_000,
                    help="write <out>.partial every N steps so an "
                         "interrupted run is not a total loss; 0 "
                         "disables")
    ap.add_argument("--entropy-final", type=float, default=None,
                    help="linearly anneal entropy_coef to this value. The "
                         "tournament loads checkpoints with sample=True, so "
                         "a policy that stays diffuse pays a sampling tax at "
                         "evaluation time; annealing sharpens it late "
                         "without losing the exploration that the early "
                         "phase needs.")
    args = ap.parse_args()

    pool = install_pool(args.pool)
    print(f"opponent pool ({args.pool}): {', '.join(sorted(pool))}")

    theory, source = load_theory(args.theory)
    print(f"theory: {theory.name} (+{theory.extra_feature_dim} features)")

    cfg = selfplay.PPOConfig(
        total_steps=args.steps, seed=args.seed, lr=args.lr,
        entropy_coef=args.entropy, p_self=args.p_self,
        p_scripted=args.p_scripted)
    trainer = selfplay.SelfPlayTrainer(theory, cfg)
    if args.resume:
        trainer.load_nets(args.resume)

    steps = []
    if args.entropy_final is not None:
        e0, e1, total = args.entropy, args.entropy_final, float(args.steps)

        def anneal(tr):
            frac = min(1.0, tr.total_steps / total)
            tr.cfg.entropy_coef = e0 + (e1 - e0) * frac

        steps.append(anneal)
        print(f"entropy anneal: {e0} -> {e1}")

    if args.save_every > 0:
        # A 700k-step run is ~1 hour, and this one has now been killed twice
        # by things that had nothing to do with the training. Only saving at
        # the end means an interruption at 95% is worth exactly as much as
        # one at 0%. The partial file is written under a separate name so a
        # crash mid-write can never corrupt a finished checkpoint.
        part = args.out + ".partial"
        state = {"next": args.save_every}

        def periodic(tr):
            if tr.total_steps >= state["next"]:
                state["next"] += args.save_every
                trainer.save_checkpoint(part, theory_source=source,
                                        team=args.team)
                print(f"  [partial saved at {tr.total_steps} steps -> {part}]",
                      flush=True)

        steps.append(periodic)

    def cb(tr):
        for fn in steps:
            fn(tr)

    trainer.train(callback=cb if steps else None)
    trainer.save_checkpoint(args.out, theory_source=source, team=args.team)


if __name__ == "__main__":
    main()
