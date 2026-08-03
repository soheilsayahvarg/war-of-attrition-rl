#!/usr/bin/env python3
"""Pick the submission checkpoint.

Scores every candidate in the OFFICIAL evaluation setting -- exactly what
`scripts/evaluate.py` does: the candidate alone against the two public
references. That matters because the score is not a property of the
candidate alone: an exploitable agent lets TitForTat farm the outlasting
prize, which raises TitForTat's number as much as it lowers yours. So we
report both, and rank by the gap.

    python analysis/select.py --episodes 50
"""
import argparse
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent           # noqa: E402
from see.tournament.runner import run_tournament            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--patterns", nargs="*",
                    default=["runs/*.pt", "runs/stock_pool/*.pt",
                             "runs/shipped_defaults/*.pt"])
    a = ap.parse_args()

    rows = []
    for pat in a.patterns:
        tag = ("" if pat == "runs/*.pt"
               else " [" + pathlib.Path(pat).parent.name + "]")
        for p in sorted(glob.glob(str(ROOT / pat))):
            name = pathlib.Path(p).stem + tag
            try:
                ag = CheckpointAgent(p)
            except Exception as e:
                print(f"  skip {name}: {e}")
                continue
            r = run_tournament({name: ag}, episodes_per_side=a.episodes,
                               seed0=a.seed0, verbose=False)
            me = next(x for x in r["leaderboard"] if x["name"] == name)
            tft = next(x for x in r["leaderboard"] if "TitForTat" in x["name"])
            rnd = next(x for x in r["leaderboard"] if "Random" in x["name"])
            rows.append({"name": name, "path": p, "score": me["score"],
                         "as_iran": me["score_as_iran"],
                         "as_us": me["score_as_us"],
                         "tft": tft["score"], "rnd": rnd["score"],
                         "edge": round(me["score"] - tft["score"], 1)})
            print(f"  {name:34s} me={me['score']:7.1f}  TFT={tft['score']:7.1f}"
                  f"   edge={rows[-1]['edge']:+7.1f}", flush=True)

    rows.sort(key=lambda r: -r["edge"])
    print(f"\n{'='*84}\nranked by edge over TitForTat (official 3-agent pool)"
          f"\n{'='*84}")
    print(f"  {'checkpoint':34s} {'score':>8s} {'as Iran':>9s} {'as U.S.':>9s} "
          f"{'TitForTat':>10s} {'edge':>8s}")
    for r in rows:
        star = "  <== BEST" if r is rows[0] else ""
        print(f"  {r['name']:34s} {r['score']:8.1f} {r['as_iran']:9.1f} "
              f"{r['as_us']:9.1f} {r['tft']:10.1f} {r['edge']:+8.1f}{star}")

    (ROOT / "analysis" / "pick_submission.json").write_text(json.dumps(rows, indent=2))
    if rows:
        print(f"\nbest: {rows[0]['path']}")


if __name__ == "__main__":
    main()
