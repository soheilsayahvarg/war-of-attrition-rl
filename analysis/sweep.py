#!/usr/bin/env python3
"""Sweep of simple parametric strategies in the OFFICIAL evaluation setting.

`scripts/evaluate.py` scores a submission against exactly two references,
Random and TitForTat, and the project manual makes "beat TitForTat" the
headline implementation criterion. That score is not a property of the
submission alone: by choosing a posture you also set how much the
references earn against you. This sweep measures both numbers.

Each strategy is (sigma, exit rule):
  sigma  -- the constant signal level it holds
  thr    -- exit once own e_hat falls below thr (thr = 0 -> never exit
            voluntarily)

    python analysis/sweep.py [--episodes 40]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.base import Agent                        # noqa: E402
from see.config import ACTION_EXIT, SIGNAL_LEVELS        # noqa: E402
from see.tournament.runner import run_tournament         # noqa: E402

E_HAT, SIG_OPP, MED = 3, 7, 8


class Const(Agent):
    def __init__(self, sigma, thr, mirror_cap=None):
        self.sigma, self.thr, self.cap = sigma, thr, mirror_cap
        self.name = f"s{sigma}_x{thr}" + (f"_cap{mirror_cap}" if mirror_cap
                                          else "")

    def act(self, obs, legal_mask, public_state=None):
        if legal_mask[ACTION_EXIT] and self.thr > 0 and obs[E_HAT] < self.thr:
            return ACTION_EXIT
        s = self.sigma
        if self.cap is not None:                # match opponent, capped
            s = min(self.cap, max(self.sigma, obs[SIG_OPP]))
        a = int(np.argmin([abs(s - x) for x in SIGNAL_LEVELS]))
        return a if legal_mask[a] else self.first_legal(legal_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=7)
    a = ap.parse_args()

    grid = []
    for sig in (0.0, 0.25, 0.5, 0.75, 1.0):
        for thr in (0.0, 0.10, 0.20, 0.35):
            grid.append(Const(sig, thr))
    for cap in (0.5, 0.75, 1.0):
        for thr in (0.0, 0.20):
            grid.append(Const(0.0, thr, mirror_cap=cap))

    rows = []
    for ag in grid:
        r = run_tournament({ag.name: ag}, episodes_per_side=a.episodes,
                           seed0=a.seed0, verbose=False)
        me = next(x for x in r["leaderboard"] if x["name"] == ag.name)
        tft = next(x for x in r["leaderboard"] if "TitForTat" in x["name"])
        rnd = next(x for x in r["leaderboard"] if "Random" in x["name"])
        rows.append({"name": ag.name, "score": me["score"],
                     "as_iran": me["score_as_iran"],
                     "as_us": me["score_as_us"],
                     "tft": tft["score"], "rnd": rnd["score"],
                     "edge": me["score"] - tft["score"]})
        print(f"  {ag.name:16s} me={me['score']:7.1f}  "
              f"(I {me['score_as_iran']:6.1f} / U {me['score_as_us']:6.1f})   "
              f"TFT={tft['score']:6.1f}   edge={rows[-1]['edge']:+7.1f}",
              flush=True)

    rows.sort(key=lambda r: -r["edge"])
    print(f"\n{'='*78}\nranked by edge over TitForTat in the same run\n{'='*78}")
    print(f"  {'strategy':16s} {'score':>8s} {'as Iran':>9s} {'as U.S.':>9s} "
          f"{'TitForTat':>10s} {'edge':>8s}")
    for r in rows:
        print(f"  {r['name']:16s} {r['score']:8.1f} {r['as_iran']:9.1f} "
              f"{r['as_us']:9.1f} {r['tft']:10.1f} {r['edge']:+8.1f}")

    (ROOT / "analysis" / "sweep.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
