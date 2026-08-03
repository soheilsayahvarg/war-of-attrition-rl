#!/usr/bin/env python3
"""Fast per-seat split (A,B,b',J_I / C,D,d',J_U) for one or more checkpoints,
plus the voluntary-exit rate of each seat against TitForTat -- which is the
behaviour the (S4) channel is supposed to change.

    python analysis/seat_split.py runs/a.pt runs/b.pt [--episodes 60]
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent                  # noqa: E402
from see.agents.scripted import RandomAgent, TitForTatAgent        # noqa: E402
from see.config import ACTION_EXIT, IRAN, US                       # noqa: E402
from see.env import StrategicEnduranceEnv                          # noqa: E402


def duel(env, iran, us, seeds, watch):
    uI, uU, ln, vex = [], [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        for role, ag in ((IRAN, iran), (US, us)):
            ag.reset(role, seed=s * 7919 + (1 if role == IRAN else 2))
        done, left = False, False
        while not done:
            public = env.public_state()
            acts = {}
            for role, ag in ((IRAN, iran), (US, us)):
                mask = env.get_legal_mask(role)
                a = ag.act(obs[role], mask, public)
                if not mask[int(a)]:
                    a = int(np.argmax(mask))
                acts[role] = int(a)
            if acts[watch] == ACTION_EXIT:
                left = True
            obs, rew, term, trunc, info = env.step(acts)
            done = term[IRAN] or trunc[IRAN]
        u = info["_true_state"]["utility"]
        uI.append(u[IRAN]); uU.append(u[US]); ln.append(env.t); vex.append(left)
    return (float(np.mean(uI)), float(np.mean(uU)), float(np.mean(ln)),
            float(np.mean(vex)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=7)
    a = ap.parse_args()

    seeds = [a.seed0 + k for k in range(a.episodes)]
    env = StrategicEnduranceEnv()
    R1, _, _, _ = duel(env, TitForTatAgent(), RandomAgent(), seeds, IRAN)
    _, R2, _, _ = duel(env, RandomAgent(), TitForTatAgent(), seeds, US)
    print("target J_I + J_U > %.1f\n" % (R1 + R2))
    print("%-22s %7s %7s %7s %8s | %7s %7s %7s %8s | %6s %6s"
          % ("checkpoint", "A", "B", "b'", "J_I",
             "C", "D", "d'", "J_U", "exI", "exU"))
    for p in a.checkpoints:
        ag = CheckpointAgent(p)
        A, _, _, _ = duel(env, ag, RandomAgent(), seeds, IRAN)
        B, bp, _, exI = duel(env, ag, TitForTatAgent(), seeds, IRAN)
        _, C, _, _ = duel(env, RandomAgent(), ag, seeds, US)
        dp, D, _, exU = duel(env, TitForTatAgent(), ag, seeds, US)
        JI, JU = A + B - bp, C + D - dp
        print("%-22s %7.1f %7.1f %7.1f %8.1f | %7.1f %7.1f %7.1f %8.1f "
              "| %6.2f %6.2f"
              % (pathlib.Path(p).name, A, B, bp, JI, C, D, dp, JU, exI, exU),
              flush=True)


if __name__ == "__main__":
    main()
