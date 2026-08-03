#!/usr/bin/env python3
"""Head-to-head comparison of every checkpoint we trained.

Runs one round robin containing all submitted variants plus the two public
reference agents, on a common seed list, and prints the leaderboard. Also
dumps per-checkpoint behavioural probes so the ablation table in the report
(theory on/off, shaping on/off, shipped vs documented hyper-parameters) can
be filled in from one command.

    python analysis/compare.py --episodes 60
"""
import argparse
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent            # noqa: E402
from see.tournament.runner import run_tournament             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--glob", default="runs/*.pt")
    ap.add_argument("--also", default="runs/shipped_defaults/*.pt")
    ap.add_argument("--out", default="analysis/compare.json")
    a = ap.parse_args()

    entrants = {}
    for pat, tag in ((a.glob, ""), (a.also, " [shipped-defaults]")):
        for p in sorted(glob.glob(str(ROOT / pat))):
            name = pathlib.Path(p).stem + tag
            try:
                entrants[name] = CheckpointAgent(p)
            except Exception as e:                        # pragma: no cover
                print(f"  skip {name}: {e}")
    print(f"entrants: {len(entrants)}  "
          f"({a.episodes} episodes per ordered pair)\n")

    res = run_tournament(entrants, episodes_per_side=a.episodes,
                         seed0=a.seed0, verbose=False)
    print(f"{'rank':>4s}  {'agent':34s} {'score':>8s} {'as Iran':>9s} "
          f"{'as U.S.':>9s}   95% CI")
    for r in res["leaderboard"]:
        print(f"{r['rank']:>4d}  {r['name']:34s} {r['score']:8.1f} "
              f"{r['score_as_iran']:9.1f} {r['score_as_us']:9.1f}   "
              f"[{r['ci95'][0]:.1f}, {r['ci95'][1]:.1f}]")

    tft = next(r for r in res["leaderboard"]
               if "TitForTat" in r["name"])
    print(f"\nbar to clear (TitForTat): {tft['score']:.1f}")
    for r in res["leaderboard"]:
        if r["name"].startswith(("full", "null", "feat", "shape")):
            d = r["score"] - tft["score"]
            print(f"  {r['name']:34s} {d:+8.1f} vs TitForTat"
                  f"{'   <-- beats it' if d > 0 else ''}")

    out = ROOT / a.out
    slim = {k: v for k, v in res.items() if k != "_episodes"}
    out.write_text(json.dumps(slim, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
