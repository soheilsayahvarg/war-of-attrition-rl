#!/usr/bin/env python3
"""Instructor tool: run the class tournament and produce the leaderboard.

    python scripts/run_tournament.py --submissions submissions/ \
        --episodes 100 --out-dir tournament_out/

Collect all student .pt files into one folder. Every submission plays every
other submission and every scripted baseline, both roles, common seeds.
Produces results.json + leaderboard.html. Use a DIFFERENT --seed0 for the
final official run than the one published for practice, so agents cannot
overfit the seed list.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from see.tournament.leaderboard import generate                 # noqa: E402
from see.tournament.runner import (load_entries, run_tournament,  # noqa: E402
                                   save_results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", default="submissions",
                    help="folder of student .pt checkpoints")
    ap.add_argument("--episodes", type=int, default=100,
                    help="episodes per side per pairing")
    ap.add_argument("--seed0", type=int, default=20_260_713)
    ap.add_argument("--out-dir", default="tournament_out")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--names-from-files", action="store_true",
                    help="name entrants by checkpoint filename (platform "
                    "uses this so board names = authenticated IDs)")
    ap.add_argument("--title", default="Strategic Endurance — "
                    "Class Tournament")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    entries = {}
    if pathlib.Path(args.submissions).is_dir():
        entries = load_entries(args.submissions,
                               prefer_filename=args.names_from_files)
    print(f"loaded {len(entries)} submissions: {sorted(entries)}")

    results = run_tournament(entries,
                             episodes_per_side=args.episodes,
                             seed0=args.seed0,
                             include_baselines=not args.no_baselines)
    save_results(results, str(out / "results.json"))
    generate(results, str(out / "leaderboard.html"), title=args.title)


if __name__ == "__main__":
    main()
