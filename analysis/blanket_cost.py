#!/usr/bin/env python3
"""How much does it cost to refuse to concede *to everybody*?

The conditional version of the override (analysis/denial_probe.py) only fires
on an identified mirror, so it is free in a large pool -- but a learned policy
cannot dodge-proof a detector that fires later than it acts.  A blanket rule
("the U.S. seat never concedes") is dodge-proof by construction.  This script
measures what it costs, opponent by opponent, so the choice is made on numbers
rather than on the argument I sketched.

    python analysis/blanket_cost.py [--episodes 60]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.base import Agent                                  # noqa: E402
from see.agents.checkpoint import CheckpointAgent                  # noqa: E402
from see.agents.scripted import RandomAgent, TitForTatAgent        # noqa: E402
from see.config import ACTION_EXIT, ACTION_HOLD, IRAN, US          # noqa: E402
from see.config import SIGNAL_LEVELS                               # noqa: E402
from see.env import StrategicEnduranceEnv                          # noqa: E402
from see.tournament.runner import run_tournament                   # noqa: E402


class NeverConcede(Agent):
    """Checkpoint policy, but the U.S. seat is not allowed to quit."""

    def __init__(self, path, hold_sigma=0.0, name="ME+blanket"):
        self.inner = CheckpointAgent(path, sample=True)
        self.hold_sigma = hold_sigma
        self.name = name

    def reset(self, player_id, seed=None):
        self.pid = player_id
        self.inner.reset(player_id, seed)

    def act(self, obs, legal_mask, public_state=None):
        a = self.inner.act(obs, legal_mask, public_state)
        if self.pid == US and a == ACTION_EXIT:
            want = int(np.argmin([abs(self.hold_sigma - s)
                                  for s in SIGNAL_LEVELS]))
            for c in (want, ACTION_HOLD, want + 1, want - 1):
                if 0 <= c < len(legal_mask) and legal_mask[c] \
                        and c != ACTION_EXIT:
                    return c
            return int(np.argmax(legal_mask))
        return a


def duel(env, iran, us, seeds):
    uI, uU = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        for role, ag in ((IRAN, iran), (US, us)):
            ag.reset(role, seed=s * 7919 + (1 if role == IRAN else 2))
        done = False
        while not done:
            public = env.public_state()
            acts = {}
            for role, ag in ((IRAN, iran), (US, us)):
                mask = env.get_legal_mask(role)
                a = ag.act(obs[role], mask, public)
                if not mask[int(a)]:
                    a = int(np.argmax(mask))
                acts[role] = int(a)
            obs, rew, term, trunc, info = env.step(acts)
            done = term[IRAN] or trunc[IRAN]
        u = info["_true_state"]["utility"]
        uI.append(u[IRAN]); uU.append(u[US])
    return float(np.mean(uI)), float(np.mean(uU))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--me", default=str(ROOT / "submission.pt"))
    a = ap.parse_args()

    seeds = [a.seed0 + k for k in range(a.episodes)]
    env = StrategicEnduranceEnv()

    plain = CheckpointAgent(a.me)
    field = [("Random", RandomAgent()), ("TitForTat", TitForTatAgent())]
    for nm, p in [("master", ROOT / "benchmarks" / "master_agent.pt"),
                  ("refspec_s3", ROOT / "runs" / "refspec_s3.pt"),
                  ("rich_s0", ROOT / "runs" / "rich_s0.pt"),
                  ("rich_s1", ROOT / "runs" / "rich_s1.pt"),
                  ("richshape_s0", ROOT / "runs" / "richshape_s0.pt")]:
        if p.exists():
            field.append((nm, CheckpointAgent(str(p))))

    print("U.S. seat, opponent by opponent   (D = our payoff, d' = theirs)")
    print("%-14s | %7s %7s | %7s %7s | %7s %7s"
          % ("Iran opponent", "D base", "d' base", "D hold0", "d' hold0",
             "dD", "dd'"))
    rows = []
    for nm, opp in field:
        dp0, D0 = duel(env, opp, plain, seeds)
        best = None
        for hs in (0.0, 0.75):
            nc = NeverConcede(a.me, hold_sigma=hs)
            dp1, D1 = duel(env, opp, nc, seeds)
            if best is None or (D1 - dp1) > (best[1] - best[0]):
                best = (dp1, D1, hs)
        dp1, D1, hs = best
        rows.append({"opp": nm, "D0": D0, "dp0": dp0, "D1": D1, "dp1": dp1,
                     "hold": hs})
        print("%-14s | %7.1f %7.1f | %7.1f %7.1f | %+7.1f %+7.1f"
              % (nm, D0, dp0, D1, dp1, D1 - D0, dp1 - dp0), flush=True)

    n = len(rows)
    dD = sum(r["D1"] - r["D0"] for r in rows) / n
    print("\nour U.S. average moves by %+.1f  => our total by %+.1f"
          % (dD, dD / 2))
    for r in rows:
        loss = (r["dp1"] - r["dp0"]) / n / 2
        print("  %-14s total moves by %+6.1f   (net vs us: %+6.1f)"
              % (r["opp"], loss, dD / 2 - loss))

    (ROOT / "analysis" / "blanket_cost.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
