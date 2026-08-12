#!/usr/bin/env python3
"""Rank candidates against a *fixed* opponent set, not a round robin.

WHY NOT analysis/official.py
----------------------------
That script runs one round robin over the whole field, which is the right
shape for the graded tournament but the wrong shape for choosing between
candidates. In a round robin the entrants are also each other's opponents,
so adding a candidate changes every other candidate's score, and two
near-identical variants (a checkpoint and a rescaled copy of it) each drag
the other's number around. The 18-entrant run and the 14-entrant run in this
project already disagree by 2-3 points on the same checkpoints for exactly
that reason.

Here the opponent set is held fixed and candidates never meet each other, so
every candidate is scored on identical episodes against identical opponents
and the differences are attributable. TitForTat is scored as a candidate too
-- it is the bar the assignment actually sets -- and is dropped from its own
opponent set so no one gets a mirror match the others do not.

The opponent set is the graded tournament's scripted pool (Random,
TitForTat, the instructor benchmark) plus behaviourally diverse checkpoints
standing in for classmates' learned agents.

    python analysis/head2head.py --episodes 40 --seed0 31415926
"""
import argparse
import io
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent               # noqa: E402
from see.agents.scripted import RandomAgent, TitForTatAgent     # noqa: E402
from see.config import IRAN, US                                 # noqa: E402
from see.env import StrategicEnduranceEnv                       # noqa: E402
from see.tournament.runner import play_episode                  # noqa: E402

MASTER = "benchmarks/master_agent.pt"

# stand-ins for the rest of the class: RL agents from this pipeline, chosen
# for behavioural spread (one never exits, one exits ~70% by stage 2, one is
# patient, one is the practice-board winner)
OPP_CKPTS = [
    ("bh_s2", "runs/bh_s2.pt"),
    ("final_v4", "runs/final_v4.pt"),
    ("bq_s1", "runs/bq_s1.pt"),
    ("bz_s1", "runs/bz_s1.pt"),
    ("ch_s3", "runs/ch_s3.pt"),
    ("master", MASTER),
]

# The opponent set is not a detail -- it changed the answer. Ranked against
# OPP_CKPTS above, the og_s2-Iran graft beats of_s3 by 9.5; ranked in the
# full round robin of analysis/official.py, of_s3 beats it by 2.7. The
# difference is entirely which generation of agent is in the room: og_s2's
# Iran net is worth 128.3 against the older checkpoints and 121.2 against
# the official-pool family, while of_s3's own Iran net is the better of the
# two against that family.
#
# Neither set is the class. `mixed` is the honest default for choosing what
# to submit: whatever wins there is not winning because of an assumption
# about which generation the classmates trained.
OPP_SETS = {
    "old": [n for n, _ in OPP_CKPTS],
    "official": ["of_s1", "of_s2", "of_s4", "og_s1", "oh_s1", "oh_s2",
                 "master"],
    "mixed": ["bh_s2", "final_v4", "bq_s1", "ch_s3", "of_s1", "of_s4",
              "og_s1", "oh_s2", "master"],
}
EXTRA_OPPS = {
    "of_s1": "runs/of_s1.pt", "of_s2": "runs/of_s2.pt",
    "of_s4": "runs/of_s4.pt", "og_s1": "runs/og_s1.pt",
    "oh_s1": "runs/oh_s1.pt", "oh_s2": "runs/oh_s2.pt",
}

CANDIDATES = [
    ("TitForTat", None),                       # the bar, scored like the rest
    ("of_s3", "runs/of_s3.pt"),
    ("of_s3_t1.5", "runs/of_s3_t1p5.pt"),
    ("of_s3_t2", "runs/of_s3_t2.pt"),
    ("of_s3_t3", "runs/of_s3_t3.pt"),
    ("of_s3_t5", "runs/of_s3_t5.pt"),
    ("of_s3_t8", "runs/of_s3_t8.pt"),
    ("of_s3_t15", "runs/of_s3_t15.pt"),
    ("of_s3_t30", "runs/of_s3_t30.pt"),
    ("bh_s2", "runs/bh_s2.pt"),
    ("bh_s2_t3", "runs/bh_s2_t3.pt"),
    ("bh_s2_t5", "runs/bh_s2_t5.pt"),
    ("of_s4", "runs/of_s4.pt"),
    ("of_s4_t5", "runs/of_s4_t5.pt"),
    ("oh_s2", "runs/oh_s2.pt"),
    ("oh_s2_t5", "runs/oh_s2_t5.pt"),
    ("og_s2", "runs/og_s2.pt"),
    # seat grafts: Iran net and U.S. net taken from different runs
    ("gr_a", "runs/gr_a.pt"),
    ("gr_b", "runs/gr_b.pt"),
    ("gr_c", "runs/gr_c.pt"),
    ("gr_e", "runs/gr_e.pt"),
    ("gr_f", "runs/gr_f.pt"),
    ("gr_g", "runs/gr_g.pt"),
    ("gr_h", "runs/gr_h.pt"),
    ("gr_i", "runs/gr_i.pt"),
    ("gr_j", "runs/gr_j.pt"),
]

# `--only` restricts the candidate list without editing the file, so a
# follow-up sweep on a second seed base can re-score a shortlist instead of
# repeating the whole table.


def opponent_factories(which="old"):
    """The two graded scripted references, plus classmate stand-ins."""
    f = {"Random": RandomAgent, "TitForTat": TitForTatAgent}
    paths = dict(OPP_CKPTS)
    paths.update(EXTRA_OPPS)
    paths["master"] = MASTER
    for name in OPP_SETS[which]:
        p = ROOT / paths[name]
        if p.exists():
            f[name] = (lambda q=str(p): CheckpointAgent(q))
        else:
            sys.stderr.write("skipping missing %s\n" % paths[name])
    return f


def score(agent_fac, opps, seeds, skip=None):
    """Mean utility over the opponent set, in both seats."""
    env = StrategicEnduranceEnv()
    me = agent_fac()
    per = {}
    for name, fac in sorted(opps.items()):
        if name == skip:            # never let anyone play a mirror match
            continue
        as_I = [play_episode(env, {IRAN: me, US: fac()}, s)["u"][IRAN]
                for s in seeds]
        as_U = [play_episode(env, {IRAN: fac(), US: me}, s)["u"][US]
                for s in seeds]
        per[name] = (float(np.mean(as_I)), float(np.mean(as_U)))
    mi = float(np.mean([v[0] for v in per.values()]))
    mu = float(np.mean([v[1] for v in per.values()]))
    return 0.5 * (mi + mu), mi, mu, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=31_415_926)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "head2head.txt"))
    ap.add_argument("--only", nargs="+", default=None,
                    help="candidate names to score; default is all of them")
    ap.add_argument("--opps", choices=tuple(OPP_SETS), default="old",
                    help="which generation of stand-ins to face")
    a = ap.parse_args()

    seeds = [a.seed0 + k for k in range(a.episodes)]
    opps = opponent_factories(a.opps)

    cands = CANDIDATES
    if a.only:
        want = set(a.only)
        cands = [c for c in CANDIDATES if c[0] in want]

    rows = []
    for name, rel in cands:
        if rel is None:
            fac, skip = TitForTatAgent, "TitForTat"
        else:
            p = ROOT / rel
            if not p.exists():
                sys.stderr.write("skipping missing %s\n" % rel)
                continue
            fac, skip = (lambda q=str(p): CheckpointAgent(q)), None
        s, mi, mu, per = score(fac, opps, seeds, skip=skip)
        rows.append((name, s, mi, mu, per))
        print("  scored %-12s %6.1f" % (name, s), flush=True)

    rows.sort(key=lambda r: -r[1])
    out = ["fixed-opponent ranking -- %d opponents (%s), %d episodes/seat, "
           "seed0=%d" % (len(opps), a.opps, a.episodes, a.seed0),
           "candidates never play each other, so these numbers are "
           "directly comparable", ""]
    out.append("%-4s %-14s %8s %9s %9s" % ("#", "candidate", "score",
                                           "as Iran", "as U.S."))
    for i, (n, s, mi, mu, _) in enumerate(rows, 1):
        mark = "   <- the bar" if n == "TitForTat" else ""
        out.append("%-4d %-14s %8.1f %9.1f %9.1f%s"
                   % (i, n, s, mi, mu, mark))

    tft = next((r for r in rows if r[0] == "TitForTat"), None)
    if tft:
        out.append("")
        out.append("margin over TitForTat:")
        for n, s, _, _, _ in rows:
            if n != "TitForTat":
                out.append("  %-14s %+6.1f" % (n, s - tft[1]))

    # per-opponent detail for the top candidate, so a win can be attributed
    top = rows[0]
    out.append("")
    out.append("per-opponent breakdown -- %s" % top[0])
    out.append("%-14s %9s %9s" % ("opponent", "as Iran", "as U.S."))
    for n, (i, u) in sorted(top[4].items()):
        out.append("%-14s %9.1f %9.1f" % (n, i, u))

    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write("\n" + text + "\n")


if __name__ == "__main__":
    main()
