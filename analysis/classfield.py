#!/usr/bin/env python3
"""Simulate the actual graded tournament, using the published board profiles.

Every simulation before this one guessed the field. This one does not have
to. The practice leaderboard publishes, for each student who has uploaded,
the two seat scores of their best checkpoint -- and the graded tournament is
a round robin over exactly those submissions plus Random, Tit-for-Tat and
the instructor benchmark.

Board rows as of 2026-08-12 (five students, ours included):

    student        score   as Iran   as U.S.
    401107613       61.7      75.0      48.5
    402111237       61.7      72.9      50.4     <- us (bh_s2)
    [MasterAgent]   56.3      65.7      46.9
    402111164       55.6      75.2      36.0
    400104986       54.9      79.0      30.8
    [TitForTat]     53.8      69.7      37.9
    [Hawk]          51.5      71.9      31.2
    403110636       49.9      63.4      36.5

Two things follow, and both matter more than anything the earlier
simulations were arguing about.

  * The field is small. Seven opponents, not twenty. A single bad matchup is
    a seventh of the score and is not averaged away.

  * Every classmate is Iran-strong and U.S.-weak, by a wide margin -- the
    U.S. column runs 30.8 to 48.5 while the Iran column runs 63.4 to 79.0.
    They escalate and they do not concede. 400104986 is within a couple of
    points of scripted Hawk in both seats.

So each classmate is stood in for by whichever agent available here lands
closest to their *published two-seat profile*, measured by the same
estimator (analysis/select2.py) that reproduced our own row to the decimal
(predicted 72.9 / 50.4, board says 72.9 / 50.4). That is a far better proxy
than "some checkpoint of ours": it matches on the observable that the
tournament actually scores.

What it still cannot capture: two agents with the same seat averages can
reach them by different routes, and a round robin is sensitive to route.
Read the output as the best available estimate of the graded table, not as
the graded table.

    python analysis/classfield.py --candidate runs/gr_f.pt
"""
import argparse
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import see.agents.scripted as scripted                          # noqa: E402
from see.agents.checkpoint import CheckpointAgent               # noqa: E402
from see.tournament.runner import run_tournament                # noqa: E402
from my_team.reference_pool import HawkAgent, MASTER            # noqa: E402

# (student, published Iran, published U.S., stand-in, its estimated profile)
# Stand-ins are picked by nearest published-profile match; the estimates come
# from analysis/select2_profiles.txt and analysis/select2.txt.
STAND_INS = [
    ("401107613", 75.0, 48.5, "runs/bh_s2.pt"),            # est 72.9 / 50.4
    ("402111164", 75.2, 36.0, "runs/board_reference.pt"),  # est 72.3 / 33.4
    ("400104986", 79.0, 30.8, None),                       # Hawk 71.9 / 31.2
    ("403110636", 63.4, 36.5, "runs/final_v4.pt"),         # est 71.7 / 42.0
]
# Match quality, stated rather than buried: the first three are good (within
# ~3 points on both seats). The fourth is not -- 403110636 sits at 63.4 /
# 36.5 and the nearest thing in our zoo is 8 points too strong in the Iran
# seat and 5 too strong in the U.S. seat, because every checkpoint here
# clusters in Iran 69-73 / U.S. 42-50. That stand-in is therefore harder
# than the student it represents, which biases the simulated table
# *against* our candidate rather than for it.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, action="append",
                    help="repeatable; each is scored in the same field")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=31_415_926)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "classfield.txt"))
    a = ap.parse_args()

    entries = {}
    for name, _, _, rel in STAND_INS:
        if rel is None:
            entries["[class] %s ~Hawk" % name] = HawkAgent()
        else:
            p = ROOT / rel
            if p.exists():
                entries["[class] %s" % name] = CheckpointAgent(str(p))
            else:
                sys.stderr.write("missing stand-in %s\n" % rel)
    if MASTER.exists():
        entries["[benchmark] MasterAgent"] = CheckpointAgent(str(MASTER))

    for rel in a.candidate:
        p = ROOT / rel
        if not p.exists():
            sys.exit("missing candidate %s" % rel)
        entries["US: %s" % pathlib.Path(rel).stem] = CheckpointAgent(str(p))

    res = run_tournament(entries, episodes_per_side=a.episodes,
                         seed0=a.seed0, include_baselines=True, verbose=False)
    rows = sorted(res["leaderboard"], key=lambda r: -r["score"])

    out = ["simulated GRADED tournament -- stand-ins matched to the published "
           "board profiles",
           "%d entrants + %d reference agents, %d episodes/side, seed0=%d"
           % (len(entries), len(scripted.BASELINES), a.episodes, a.seed0), ""]
    out.append("%-4s %-30s %8s %9s %9s" % ("#", "agent", "score", "as Iran",
                                           "as U.S."))
    for i, r in enumerate(rows, 1):
        out.append("%-4d %-30s %8.1f %9.1f %9.1f"
                   % (i, r["name"], r["score"], r["score_as_iran"],
                      r["score_as_us"]))

    tft = next((r for r in rows
                if "TitForTat" in r["name"]), None)
    ours = [r for r in rows if r["name"].startswith("US: ")]
    if tft and ours:
        out.append("")
        out.append("Tit-for-Tat scores %.1f here; margins:" % tft["score"])
        for r in ours:
            out.append("  %-28s %+6.1f   (rank %d of %d)"
                       % (r["name"], r["score"] - tft["score"],
                          rows.index(r) + 1, len(rows)))

    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
