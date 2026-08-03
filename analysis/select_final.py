#!/usr/bin/env python3
"""Score every candidate checkpoint on the criterion that is actually graded.

Two measurements per candidate:

  reference : the pool scripts/evaluate.py builds, {you, Random, TitForTat}
  class     : the same plus every other checkpoint we have, as a stand-in
              for the class round robin

For each we report our own score, TitForTat's score in the *same* run, and
the edge between them.  A candidate is only preferred if it wins the edge in
the reference pool without giving up own score in the class pool.

    python analysis/select_final.py runs/deny_s11.pt runs/deny_s12.pt ...
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent            # noqa: E402
from see.tournament.runner import run_tournament             # noqa: E402


def measure(agent, field, episodes, seed0):
    entries = {"ME": agent}
    entries.update(field)
    r = run_tournament(entries, episodes_per_side=episodes, seed0=seed0,
                       verbose=False)
    b = {row["name"]: row for row in r["leaderboard"]}
    me, tft = b["ME"], next(v for k, v in b.items() if "TitForTat" in k)
    return {"me": me["score"], "me_I": me["score_as_iran"],
            "me_U": me["score_as_us"], "tft": tft["score"],
            "edge": me["score"] - tft["score"], "rank": me["rank"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=7)
    a = ap.parse_args()

    field = {}
    for nm, p in [("master", ROOT / "benchmarks" / "master_agent.pt"),
                  ("refspec_s3", ROOT / "runs" / "refspec_s3.pt"),
                  ("rich_s0", ROOT / "runs" / "rich_s0.pt"),
                  ("rich_s1", ROOT / "runs" / "rich_s1.pt"),
                  ("richshape_s0", ROOT / "runs" / "richshape_s0.pt")]:
        if p.exists():
            field[nm] = CheckpointAgent(str(p))

    print("%-26s | %-28s | %-28s" % ("", "reference pool (graded)",
                                     "class-sized pool"))
    print("%-26s | %7s %7s %7s | %7s %7s %7s" %
          ("candidate", "me", "TFT", "edge", "me", "TFT", "edge"))
    print("-" * 88)

    rows = []
    for path in a.candidates:
        try:
            ag = CheckpointAgent(path)
        except Exception as e:                               # noqa: BLE001
            print("%-26s | load failed: %s" % (pathlib.Path(path).name, e))
            continue
        ref = measure(ag, {}, a.episodes, a.seed0)
        cls = measure(ag, field, a.episodes, a.seed0)
        rows.append({"path": path, "name": pathlib.Path(path).name,
                     "ref": ref, "cls": cls})
        print("%-26s | %7.1f %7.1f %+7.1f | %7.1f %7.1f %+7.1f" %
              (pathlib.Path(path).name, ref["me"], ref["tft"], ref["edge"],
               cls["me"], cls["tft"], cls["edge"]), flush=True)

    rows.sort(key=lambda r: (-(r["ref"]["edge"] > 0), -r["cls"]["edge"],
                             -r["ref"]["edge"]))
    print("\nranking (must win the reference edge first, then class edge):")
    for i, r in enumerate(rows, 1):
        print("  %d. %-24s ref %+7.1f   class %+7.1f   own %6.1f"
              % (i, r["name"], r["ref"]["edge"], r["cls"]["edge"],
                 r["cls"]["me"]))
    if rows:
        print("\n=> pick: " + rows[0]["path"])
    (ROOT / "analysis" / "select_final.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
