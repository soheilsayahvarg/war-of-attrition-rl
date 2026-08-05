#!/usr/bin/env python3
"""How much of our board score is the agent, and how much is the seed list?

The practice board publishes its seed base (20260901). DESIGN.md Sec. 10.4
says the official run uses a different, undisclosed one, so the honest
question is not "what did we score" but "what is the spread of that score
across seed bases we did not tune on".

This replays the score against the four references that are *exactly*
reproduced locally -- Random, TitForTat, Hawk, MasterAgent -- on several
disjoint seed bases. Those four are the trustworthy part of the replica
(they match the board to the decimal on both uploads), so their spread is
a clean estimate of seed sensitivity, uncontaminated by reconstruction
error in the two soft slots.

    python analysis/seed_shift.py submission.pt [--blocks 5]
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
from see.config import IRAN, US                                # noqa: E402
from see.env import StrategicEnduranceEnv                      # noqa: E402
from see.tournament.runner import play_episode                 # noqa: E402
from my_team.reference_pool import HawkAgent, MASTER           # noqa: E402
from see.agents.checkpoint import CheckpointAgent as _CA       # noqa: E402

EXACT = {
    "Random": RandomAgent,
    "TitForTat": TitForTatAgent,
    "Hawk": HawkAgent,
    "MasterAgent": lambda: _CA(str(MASTER)),
}


def block(me, seeds, env):
    per = {}
    for nm, fac in EXACT.items():
        i = np.mean([play_episode(env, {IRAN: me, US: fac()}, s)["u"][IRAN]
                     for s in seeds])
        u = np.mean([play_episode(env, {IRAN: fac(), US: me}, s)["u"][US]
                     for s in seeds])
        per[nm] = (float(i), float(u))
    mi = float(np.mean([v[0] for v in per.values()]))
    mu = float(np.mean([v[1] for v in per.values()]))
    return per, mi, mu, 0.5 * (mi + mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "seed_shift.txt"))
    a = ap.parse_args()

    # the published base, then bases we never looked at while tuning
    bases = [20_260_901] + [31_000_000 + 104_729 * k
                            for k in range(a.blocks - 1)]
    env = StrategicEnduranceEnv()
    out = ["score on the four exactly-reproduced references, by seed base",
           "(%d episodes/role; base 20260901 is the published practice one)"
           % a.episodes, ""]
    out.append("%-22s %s %9s %8s" % ("checkpoint",
                                     " ".join("%9d" % b for b in bases),
                                     "mean", "spread"))
    for c in a.ckpts:
        me = CheckpointAgent(str(ROOT / c))
        sc = []
        for b in bases:
            seeds = [b + k for k in range(a.episodes)]
            sc.append(block(me, seeds, env)[3])
        out.append("%-22s %s %9.1f %8.1f"
                   % (pathlib.Path(c).stem,
                      " ".join("%9.1f" % s for s in sc),
                      float(np.mean(sc)), float(np.max(sc) - np.min(sc))))

    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
