#!/usr/bin/env python3
"""What exactly does a checkpoint DO in one seat against one opponent.

The board says our U.S. seat scores 113.3 against Random with one
checkpoint and 64.1 with another, and 64.6 against Dove with the first and
113.1 with the second. Aggregate scores cannot say why. This prints the
behaviour itself -- signal level, who exits, when, and what the episode
paid -- for a (checkpoint, seat, opponent) triple, so the two policies can
be compared move by move rather than through a mean.

    python analysis/seat_probe.py runs/final_v4.pt runs/bh_s2.pt \
        --opponents Random Dove Hawk
"""
import argparse
import io
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent              # noqa: E402
from see.agents.scripted import RandomAgent, TitForTatAgent    # noqa: E402
from see.config import (ACTION_EXIT, ACTION_HOLD, IRAN,        # noqa: E402
                        SIGNAL_LEVELS, US)
from see.env import StrategicEnduranceEnv                      # noqa: E402
from my_team.opponents import PatientAgent, PlateauAgent       # noqa: E402
from my_team.reference_pool import (BayesianThresholdAgent,    # noqa: E402
                                    DoveAgent, HawkAgent, MASTER)

OPPS = {
    "Random": RandomAgent,
    "TitForTat": TitForTatAgent,
    "Hawk": HawkAgent,
    "Dove": DoveAgent,
    "Patient": PatientAgent,
    "Plateau": PlateauAgent,
    "BayesianThreshold": BayesianThresholdAgent,
    "MasterAgent": lambda: CheckpointAgent(str(MASTER)),
}


def probe(me, fac, seat, seeds, env):
    other = US if seat == IRAN else IRAN
    sig, ln, we_exit, they_exit, timeout, u = [], [], [], [], [], []
    for s in seeds:
        agents = {seat: me, other: fac()}
        obs, _ = env.reset(seed=s)
        for pid, ag in agents.items():
            ag.reset(pid, seed=s * 7919 + (1 if pid == IRAN else 2))
        done, mine = False, []
        while not done:
            public = env.public_state()
            acts = {}
            for pid, ag in agents.items():
                mask = env.get_legal_mask(pid)
                a = int(ag.act(obs[pid], mask, public))
                if not mask[a]:
                    a = int(np.argmax(mask))
                acts[pid] = a
            mine.append(acts[seat])
            obs, _r, term, trunc, info = env.step(acts)
            done = term[IRAN] or trunc[IRAN]
        lv = [SIGNAL_LEVELS[a] for a in mine if a < len(SIGNAL_LEVELS)]
        if lv:
            sig.append(float(np.mean(lv)))
        ln.append(env.t)
        ex = {p: env.exited[p] for p in (IRAN, US)}
        we_exit.append(1.0 if ex[seat] else 0.0)
        they_exit.append(1.0 if ex[other] else 0.0)
        timeout.append(0.0 if (ex[seat] or ex[other]) else 1.0)
        u.append(info["_true_state"]["utility"][seat])
    return {
        "u": float(np.mean(u)),
        "sigma": float(np.mean(sig)) if sig else 0.0,
        "len": float(np.mean(ln)),
        "we exit": float(np.mean(we_exit)),
        "they exit": float(np.mean(they_exit)),
        "timeout": float(np.mean(timeout)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--opponents", nargs="+", default=["Random", "Dove"])
    ap.add_argument("--seat", default="U", choices=("I", "U", "both"))
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=20_260_901)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "seat_probe.txt"))
    a = ap.parse_args()

    env = StrategicEnduranceEnv()
    seeds = [a.seed0 + k for k in range(a.episodes)]
    seats = [IRAN, US] if a.seat == "both" else \
            [IRAN if a.seat == "I" else US]
    keys = ["u", "sigma", "len", "we exit", "they exit", "timeout"]
    out = ["behaviour by (checkpoint, seat, opponent) -- %d episodes"
           % a.episodes, ""]
    for seat in seats:
        for nm in a.opponents:
            out.append("seat %s   vs %s" % ("Iran" if seat == IRAN else "U.S.",
                                            nm))
            out.append("  %-16s %8s %8s %8s %9s %10s %8s"
                       % ("checkpoint", *keys))
            for c in a.ckpts:
                me = CheckpointAgent(str(ROOT / c))
                r = probe(me, OPPS[nm], seat, seeds, env)
                out.append("  %-16s %8.1f %8.3f %8.1f %9.3f %10.3f %8.3f"
                           % (pathlib.Path(c).stem, *[r[k] for k in keys]))
            out.append("")
    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
