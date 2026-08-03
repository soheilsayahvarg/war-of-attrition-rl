#!/usr/bin/env python3
"""Is the edge over TitForTat a property of the agent or of the seed list?

The manual says the official run uses fresh, undisclosed seeds, so an edge
that only exists on seed0=7 is worth nothing. This replays the reference
evaluation on several disjoint seed blocks and reports the edge on each.

    python analysis/seed_robust.py runs/a.pt [runs/b.pt ...] [--blocks 6]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent            # noqa: E402
from see.tournament.runner import run_tournament             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--blocks", type=int, default=6)
    a = ap.parse_args()

    # disjoint seed blocks, none of them the one we tuned on
    starts = [7] + [100_000 + 7919 * k for k in range(a.blocks - 1)]

    print("%-24s %s" % ("checkpoint",
                        " ".join("%8s" % ("s%d" % s) for s in starts)))
    out = []
    for p in a.checkpoints:
        ag = CheckpointAgent(p)
        edges = []
        for s0 in starts:
            r = run_tournament({"ME": ag}, episodes_per_side=a.episodes,
                               seed0=s0, verbose=False)
            b = {row["name"]: row for row in r["leaderboard"]}
            tft = next(v for k, v in b.items() if "TitForTat" in k)
            edges.append(b["ME"]["score"] - tft["score"])
        e = np.array(edges)
        print("%-24s %s   mean %+6.1f  min %+6.1f  wins %d/%d"
              % (pathlib.Path(p).name,
                 " ".join("%+8.1f" % v for v in edges),
                 e.mean(), e.min(), int((e > 0).sum()), len(e)), flush=True)
        out.append({"path": p, "starts": starts, "edges": edges,
                    "mean": float(e.mean()), "min": float(e.min()),
                    "wins": int((e > 0).sum())})

    out.sort(key=lambda r: (-r["wins"], -r["mean"]))
    print("\nmost robust first:")
    for r in out:
        print("  %-24s wins %d/%d  mean %+6.1f"
              % (pathlib.Path(r["path"]).name, r["wins"], len(r["edges"]),
                 r["mean"]))
    (ROOT / "analysis" / "seed_robust.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
