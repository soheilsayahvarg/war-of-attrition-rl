#!/usr/bin/env python3
"""Does the criterion "score above TitForTat" depend on the size of the pool?

`scripts/evaluate.py` scores you in a three-agent pool {you, Random, TFT}.
The class tournament scores you in a pool with every other submission in it.
Those are not the same measurement: TitForTat's own score is an average over
its opponents, so a single opponent that denies it the prize moves its average
a lot in a small pool and almost not at all in a large one.

This script measures the same submission in pools of growing size, using the
other checkpoints in runs/ plus the instructor benchmark as stand-ins for
classmates.

    python analysis/classpool.py [--episodes 40]
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent            # noqa: E402
from see.tournament.runner import run_tournament             # noqa: E402


def board(entries, episodes, seed0):
    r = run_tournament(entries, episodes_per_side=episodes,
                       seed0=seed0, verbose=False)
    return {row["name"]: row for row in r["leaderboard"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--me", default=str(ROOT / "submission.pt"))
    a = ap.parse_args()

    me = CheckpointAgent(a.me)
    ME = "ME"

    classmates = [
        ("master", ROOT / "benchmarks" / "master_agent.pt"),
        ("refspec_s3", ROOT / "runs" / "refspec_s3.pt"),
        ("rich_s0", ROOT / "runs" / "rich_s0.pt"),
        ("rich_s1", ROOT / "runs" / "rich_s1.pt"),
        ("richshape_s0", ROOT / "runs" / "richshape_s0.pt"),
    ]
    loaded = []
    for nm, p in classmates:
        if p.exists():
            try:
                loaded.append((nm, CheckpointAgent(str(p))))
            except Exception as e:                            # noqa: BLE001
                print(f"  !! skip {nm}: {e}")

    rows = []
    for k in range(len(loaded) + 1):
        entries = {ME: me}
        for nm, ag in loaded[:k]:
            entries[nm] = ag
        b = board(entries, a.episodes, a.seed0)
        mine = b[ME]
        tft = next(v for kk, v in b.items() if "TitForTat" in kk)
        n_pool = len(b)
        rows.append({"extra_entrants": k, "pool_size": n_pool,
                     "me": mine["score"], "me_I": mine["score_as_iran"],
                     "me_U": mine["score_as_us"],
                     "tft": tft["score"], "tft_I": tft["score_as_iran"],
                     "tft_U": tft["score_as_us"],
                     "edge": mine["score"] - tft["score"]})
        r = rows[-1]
        print(f"pool={n_pool:2d} (+{k} classmates)  "
              f"ME={r['me']:6.1f} (I {r['me_I']:6.1f}/U {r['me_U']:6.1f})   "
              f"TFT={r['tft']:6.1f} (I {r['tft_I']:6.1f}/U {r['tft_U']:6.1f})"
              f"   edge={r['edge']:+7.1f}", flush=True)

    (ROOT / "analysis" / "classpool.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
