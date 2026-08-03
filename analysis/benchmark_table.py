#!/usr/bin/env python3
"""Regenerate the two tables of report Sec. 11.8: the four-way leaderboard
against the instructor benchmark, and the six head-to-head pairings.

    python analysis/benchmark_table.py [submission.pt] [--episodes 60]
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent            # noqa: E402
from see.tournament.runner import run_tournament             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=str(ROOT / "submission.pt"))
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=7)
    a = ap.parse_args()

    me = CheckpointAgent(a.checkpoint)
    entries = {"SUBMISSION": me}
    mp = ROOT / "benchmarks" / "master_agent.pt"
    if mp.exists():
        entries["MasterAgent"] = CheckpointAgent(str(mp))

    r = run_tournament(entries, episodes_per_side=a.episodes,
                       seed0=a.seed0, verbose=False)

    print("%-30s %8s %8s %8s   %s"
          % ("agent", "score", "as Iran", "as U.S.", "95% CI"))
    for row in r["leaderboard"]:
        print("%-30s %8.1f %8.1f %8.1f   [%.1f, %.1f]"
              % (row["name"], row["score"], row["score_as_iran"],
                 row["score_as_us"], row["ci_lo"], row["ci_hi"]))

    print("\nhead to head")
    print("%-16s %-16s %8s %8s %7s" % ("Iran", "U.S.", "u_I", "u_U", "len"))
    for pr in r["pairs"]:
        if "SUBMISSION" in (pr["iran"], pr["us"]):
            print("%-16s %-16s %8.1f %8.1f %7.1f"
                  % (pr["iran"][:16], pr["us"][:16], pr["u_I"], pr["u_U"],
                     pr["len"]))

    slim = {"leaderboard": r["leaderboard"],
            "pairs": [{k: v for k, v in p.items() if not k.startswith("raw_")}
                      for p in r["pairs"]]}
    (ROOT / "analysis" / "benchmark_table.json").write_text(
        json.dumps(slim, indent=1))


if __name__ == "__main__":
    main()
