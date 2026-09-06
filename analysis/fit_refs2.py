#!/usr/bin/env python3
"""Refit Dove and BayesianThreshold against *two* checkpoints at once.

`analysis/fit_refs.py` fitted them to one upload's per-opponent row. One
checkpoint is a weak constraint: it pins the score the reference produces
against one particular opponent, and many parameter settings do that.

The second upload gave a second row, and the two rows disagree in a way
that is itself the evidence:

                     upload 1           upload 2
    real Dove        144.4 / 135.9       104.9 /  64.6
    replica Dove     154.8 / 130.6       152.8 / 143.3

The replica hands both checkpoints roughly the same score; the real Dove
punishes the second one hard. A reference that discriminates between two
of our policies is not leaving at the first opportunity -- it stays long
enough that an opponent who folds early loses the standoff. So theta is
too high, and the two rows together pin it far better than either alone.

Same story, smaller, for BayesianThreshold (77.6 / 9.4 then 60.9 / 15.0).

Fit criterion: mean absolute error over all four numbers per agent, both
checkpoints. Four exactly-known references (Random, TitForTat, Hawk,
MasterAgent) already reproduce to the decimal, so the harness itself is
not in question -- only these two agents are.

    python analysis/fit_refs2.py [--episodes 40]
"""
import argparse
import io
import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent              # noqa: E402
from see.config import IRAN, US                                # noqa: E402
from see.env import StrategicEnduranceEnv                      # noqa: E402
from see.tournament.runner import play_episode                 # noqa: E402
from my_team.reference_pool import (BayesianThresholdAgent,    # noqa: E402
                                    DoveAgent)

# checkpoint -> (as Iran, as U.S.) as reported by the board
TARGETS = {
    "Dove": {
        "runs/board_reference.pt": (144.4, 135.9),
        "submission.pt": (104.9, 64.6),
    },
    "BayesianThreshold": {
        "runs/board_reference.pt": (77.6, 9.4),
        "submission.pt": (60.9, 15.0),
    },
}


def score(me, factory, seeds, env):
    as_I = [play_episode(env, {IRAN: me, US: factory()}, s)["u"][IRAN]
            for s in seeds]
    as_U = [play_episode(env, {IRAN: factory(), US: me}, s)["u"][US]
            for s in seeds]
    return float(np.mean(as_I)), float(np.mean(as_U))


def sweep(name, configs, label, agents, seeds, env, out):
    """configs: list of (tag, factory). Returns the best tag."""
    tgt = TARGETS[name]
    out.append("%s: %s" % (name, label))
    hdr = "  %-26s" % "parameters"
    for c in tgt:
        hdr += " %19s" % pathlib.Path(c).stem
    hdr += " %8s" % "MAE"
    out.append(hdr)
    best = None
    for tag, fac in configs:
        line = "  %-26s" % tag
        errs = []
        for ck, t in tgt.items():
            i, u = score(agents[ck], fac, seeds, env)
            line += " %9.1f %9.1f" % (i, u)
            errs += [abs(i - t[0]), abs(u - t[1])]
        mae = float(np.mean(errs))
        line += " %8.1f" % mae
        out.append(line)
        if best is None or mae < best[0]:
            best = (mae, tag)
    line = "  %-26s" % "TARGET (board)"
    for c, t in tgt.items():
        line += " %9.1f %9.1f" % t
    out.append(line)
    out.append("  best: %s   (MAE %.1f)" % (best[1], best[0]))
    out.append("")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=20_260_901)
    ap.add_argument("--stage", default="dove", choices=("dove", "bt"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    env = StrategicEnduranceEnv()
    seeds = [a.seed0 + k for k in range(a.episodes)]
    agents = {c: CheckpointAgent(str(ROOT / c))
              for c in TARGETS["Dove"]}
    out = ["refit against two checkpoints -- %d episodes/role, seed0=%d"
           % (a.episodes, a.seed0), ""]

    if a.stage == "dove":
        # theta = -1 encodes "never exits voluntarily" (e_hat >= 0 always)
        cfg = [("theta=%.2f" % th, lambda th=th: DoveAgent(theta=th))
               for th in (0.95, 0.85, 0.75, 0.65, 0.55, 0.45,
                          0.35, 0.25, 0.15, -1.0)]
        sweep("Dove", cfg, "sweep the endurance threshold theta",
              agents, seeds, env, out)
    else:
        cfg = []
        for fold, pb, ef in itertools.product(
                (0.55, 0.70, 0.85), (0.25, 0.375, 0.55), (0.35, 0.55, 2.0)):
            def fac(fold=fold, pb=pb, ef=ef):
                return BayesianThresholdAgent(fold=fold, push_below=pb,
                                              e_fold=ef)
            cfg.append(("fold=%.2f push<%.3f e<%.2f" % (fold, pb, ef), fac))
        sweep("BayesianThreshold", cfg, "sweep fold / push_below / e_fold",
              agents, seeds, env, out)

    text = "\n".join(out)
    path = a.out or str(ROOT / "analysis" / ("fit2_%s.txt" % a.stage))
    io.open(path, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
