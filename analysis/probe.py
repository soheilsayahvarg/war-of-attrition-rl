#!/usr/bin/env python3
"""Behavioural probe of a checkpoint: what does the trained pair actually do?

Prints, per role, the distribution over first-stage actions and the exit
timing -- enough to spot the "concession collapse" degeneracy described in
docs/DESIGN.md Sec. 11.3 (one seat learns to exit at t ~ 0 and the other
never exits).

    python analysis/probe.py runs/full_s0.pt [--vs TitForTat]
"""
import argparse
import collections
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see import IRAN, PLAYERS, US                      # noqa: E402
from see.agents import BASELINES                        # noqa: E402
from see.agents.checkpoint import CheckpointAgent       # noqa: E402
from see.config import ACTION_EXIT, ACTION_HOLD, SIGNAL_LEVELS  # noqa: E402
from see.env import StrategicEnduranceEnv               # noqa: E402
from see.tournament.runner import play_episode          # noqa: E402

LABEL = {i: f"sigma={SIGNAL_LEVELS[i]}" for i in range(5)}
LABEL[ACTION_EXIT] = "EXIT"
LABEL[ACTION_HOLD] = "HOLD"


def probe(ckpt, opp_name, episodes, seed0=7):
    sub = CheckpointAgent(ckpt)
    opp = BASELINES[opp_name]()
    env = StrategicEnduranceEnv()
    out = {}
    for role in (IRAN, US):
        agents = ({IRAN: sub, US: opp} if role == IRAN
                  else {IRAN: opp, US: sub})
        first = collections.Counter()
        exit_t, us, lens, forced = [], [], [], 0
        for s in range(seed0, seed0 + episodes):
            r = play_episode(env, agents, s)
            h = r["history"]
            first[h[0]["action"][role]] += 1
            us.append(r["u"][role])
            lens.append(r["steps"])
            te = next((e["t"] for e in h if e["exit"][role]), None)
            if te is not None:
                exit_t.append(te)
        out[role] = {
            "first": first, "u": float(np.mean(us)),
            "len": float(np.mean(lens)),
            "exit_rate": len(exit_t) / episodes,
            "exit_t": float(np.mean(exit_t)) if exit_t else float("nan"),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--vs", default="TitForTat")
    ap.add_argument("--episodes", type=int, default=60)
    a = ap.parse_args()

    res = probe(a.checkpoint, a.vs, a.episodes)
    print(f"{a.checkpoint}   vs {a.vs}   ({a.episodes} episodes)\n")
    for role in (IRAN, US):
        d = res[role]
        tot = sum(d["first"].values())
        dist = "  ".join(
            f"{LABEL[k]}:{v/tot:.0%}"
            for k, v in sorted(d["first"].items(), key=lambda x: -x[1]))
        print(f"  as {role}:  u={d['u']:7.1f}   ep_len={d['len']:4.1f}   "
              f"exits itself {d['exit_rate']:.0%} of episodes "
              f"(mean t={d['exit_t']:.1f})")
        print(f"          first action: {dist}")
