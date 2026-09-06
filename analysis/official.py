#!/usr/bin/env python3
"""Score candidates under the *graded* criterion, not the practice board.

Two different tournaments have been conflated all along.

  practice board   5 scripted baselines + the instructor benchmark, on a
                   published seed base. This is what SEEBoard shows, and
                   everything in this project was tuned against it.

  class tournament the assignment sheet, Sec. "ارزیابی و رقابت کلاسی":
                   every submission plays every other submission, the *two*
                   reference agents, and the instructor benchmark, in both
                   roles on a common list of fresh, undisclosed seeds.

                   Note the pool size. docs/DESIGN.md Sec. 10.3 says "all
                   five scripted baselines", but that document is the v1.0
                   engine design and Sec. 11 records that the public
                   reference pool was later cut to two. The assignment sheet
                   is the authoritative student-facing text and it says two,
                   naming them: Random and Tit-for-Tat. So Hawk, Dove and
                   BayesianThreshold are in the practice board only -- they
                   are not in the graded tournament at all.

They are not the same objective. In a class of N students the official pool
is 5 scripted agents and N learned ones, so as N grows the score is
dominated by how the agent does against *other people's RL agents* -- not
against Dove and BayesianThreshold, which is what the practice board
rewards and what every checkpoint here was selected on.

There is no way to see other students' agents, so this uses our own
checkpoint zoo as stand-ins. They are a fair proxy in the one way that
matters: they are RL agents trained on this game by the same pipeline, and
they are behaviourally diverse (one pins sigma = 0.75 and never exits,
another exits 70% of the time by stage 2, another is patient).

Every entrant is scored in a single round robin, so all candidates are
ranked against an identical field in one pass.

    python analysis/official.py --episodes 40 --seed0 31415926
"""
import argparse
import io
import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import see.agents.scripted as scripted                        # noqa: E402
from see.agents.checkpoint import CheckpointAgent             # noqa: E402
from see.tournament.runner import run_tournament              # noqa: E402

# Left exactly as the repository ships it: Random and TitForTat. That is
# the graded pool, and installing the reconstructed five here would be
# simulating the practice board again by mistake.

# candidates + stand-ins for classmates. Chosen for behavioural spread, not
# for practice-board score: see analysis/seat_probe.py for how differently
# these actually play.
FIELD = [
    ("gr_b_graft", "runs/gr_b.pt"),
    ("gr_a_graft", "runs/gr_a.pt"),
    ("og_s1_official", "runs/og_s1.pt"),
    ("og_s2_official", "runs/og_s2.pt"),
    ("oh_s1_officialH", "runs/oh_s1.pt"),
    ("oh_s2_officialH", "runs/oh_s2.pt"),
    ("of_s1_official", "runs/of_s1.pt"),
    ("of_s2_official", "runs/of_s2.pt"),
    ("of_s3_official", "runs/of_s3.pt"),
    ("of_s4_official", "runs/of_s4.pt"),
    ("bh_s2_board61.7", "runs/bh_s2.pt"),
    ("bp_s4_board58.8", "runs/bp_s4.pt"),
    ("final_v4_board56.8", "runs/final_v4.pt"),
    ("board_ref_52.9", "runs/board_reference.pt"),
    ("by_s2_highscripted", "runs/by_s2.pt"),
    ("bz_s1_20feature", "runs/bz_s1.pt"),
    ("ch_s3", "runs/ch_s3.pt"),
    ("bq_s1", "runs/bq_s1.pt"),
    ("bp_s1", "runs/bp_s1.pt"),
    ("master_benchmark", "benchmarks/master_agent.pt"),
]

# The field is a guess about the class, and the guess matters: a score is a
# mean over opponents, so who else is in the room changes the ranking. Two
# stress cases bracket the plausible range.
#
#   strong  only the checkpoints that score well here -- the class if
#           everyone tunes as hard as we did. This is the hostile case for
#           us, because our margin over TitForTat comes from the U.S. seat
#           against opponents that concede, and strong agents concede less.
#   small   a six-agent room, where each individual opponent carries much
#           more weight and a single bad matchup is not averaged away.
WEAK = {"final_v4_board56.8", "board_ref_52.9", "bp_s1", "by_s2_highscripted"}
SMALL = {"of_s3_official", "bh_s2_board61.7", "oh_s2_officialH", "bq_s1",
         "bz_s1_20feature", "master_benchmark"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=31_415_926,
                    help="deliberately NOT the published practice base")
    ap.add_argument("--out", default=str(ROOT / "analysis" / "official.txt"))
    ap.add_argument("--field", choices=("all", "strong", "small"),
                    default="all")
    a = ap.parse_args()

    field = FIELD
    if a.field == "strong":
        field = [e for e in FIELD if e[0] not in WEAK]
    elif a.field == "small":
        field = [e for e in FIELD if e[0] in SMALL]

    entries = {}
    for name, rel in field:
        p = ROOT / rel
        if p.exists():
            entries[name] = CheckpointAgent(str(p))
        else:
            sys.stderr.write("skipping missing %s\n" % rel)

    res = run_tournament(entries, episodes_per_side=a.episodes,
                         seed0=a.seed0, include_baselines=True, verbose=False)

    rows = sorted(res["leaderboard"], key=lambda r: -r["score"])
    out = ["class-tournament criterion -- %d entrants (%s field) + %d "
           "reference agents"
           % (len(entries), a.field, len(scripted.BASELINES)),
           "%d episodes/side, seed0=%d (not the published practice base)"
           % (a.episodes, a.seed0), ""]
    out.append("%-4s %-22s %8s %9s %9s" % ("#", "agent", "score",
                                           "as Iran", "as U.S."))
    for i, r in enumerate(rows, 1):
        mark = ""
        if r["name"].startswith("[baseline]"):
            mark = "   <- scripted"
        out.append("%-4d %-22s %8.1f %9.1f %9.1f%s"
                   % (i, r["name"], r["score"], r["score_as_iran"],
                      r["score_as_us"], mark))

    out.append("")
    ours = [r for r in rows if not r["name"].startswith("[baseline]")]
    best = ours[0]
    out.append("best candidate under this criterion: %s (%.1f)"
               % (best["name"], best["score"]))
    board_best = next((r for r in rows if r["name"] == "bh_s2_board61.7"),
                      None)
    if board_best and board_best["name"] != best["name"]:
        out.append("the practice-board checkpoint bh_s2 (61.7) scores %.1f here"
                   % board_best["score"])
        out.append("-> the two criteria disagree; the graded one is this "
                   "table.")

    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
