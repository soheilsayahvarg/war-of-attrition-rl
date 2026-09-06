#!/usr/bin/env python3
"""Score every checkpoint's two seats against the reconstructed class field.

WHY THIS EXISTS
---------------
analysis/seatscan.py ranked the seats against a set of older checkpoints,
and analysis/classfield.py then showed that ranking does not transfer: the
Iran net that scored 128.3 against that set scored 78.9 against the field
the leaderboard says we will actually face, and the graft built on it went
from +11.2 to -7.7. So the seat scan answered the right question about the
wrong room.

This is the same scan against the right room -- the four classmate stand-ins
matched to their published two-seat profiles, the instructor benchmark, and
the two graded reference agents (see analysis/classfield.py for how the
stand-ins were chosen and where the match is weak).

Two things fall out that nothing else measures:

  * the best whole checkpoint under the graded criterion, searched over the
    *entire* run directory rather than the dozen or so that earlier screens
    happened to promote. Every checkpoint is a valid submission on its own --
    it carries its own theory.py -- so the older-theory runs are candidates
    here even though they cannot contribute a seat to a graft.

  * the best Iran net and the best U.S. net *in this field*, which is what a
graft should have been built from at the outset. Grafting is
    restricted to checkpoints sharing the current theory, since the
    checkpoint has room for only one copy of it.

Scores are means over a fixed opponent set, so candidates never play each
other and the numbers are directly comparable -- unlike a round robin, where
adding a candidate perturbs everyone else's score.

    python analysis/classscan.py --episodes 20
"""
import argparse
import hashlib
import io
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent               # noqa: E402
from see.agents.scripted import RandomAgent, TitForTatAgent     # noqa: E402
from see.config import IRAN, US                                 # noqa: E402
from see.env import StrategicEnduranceEnv                       # noqa: E402
from see.tournament.runner import play_episode                  # noqa: E402
from my_team.reference_pool import HawkAgent, MASTER            # noqa: E402
from analysis.classfield import STAND_INS                       # noqa: E402


def theory_sha(path):
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return None
    return hashlib.sha256((ck.get("theory_source") or "").encode()).hexdigest()


def build_opps():
    """The graded field, minus the candidate itself."""
    f = [("Random", RandomAgent), ("TitForTat", TitForTatAgent)]
    for student, _, _, rel in STAND_INS:
        if rel is None:
            f.append(("~Hawk(%s)" % student, HawkAgent))
        else:
            p = ROOT / rel
            if p.exists():
                f.append((student, lambda q=str(p): CheckpointAgent(q)))
    if MASTER.exists():
        f.append(("master", lambda: CheckpointAgent(str(MASTER))))
    return f


def seats(ckpt, opps, seeds, env):
    me = CheckpointAgent(str(ckpt))
    i = [np.mean([play_episode(env, {IRAN: me, US: fac()}, s)["u"][IRAN]
                  for s in seeds]) for _, fac in opps]
    u = [np.mean([play_episode(env, {IRAN: fac(), US: me}, s)["u"][US]
                  for s in seeds]) for _, fac in opps]
    return float(np.mean(i)), float(np.mean(u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=31_415_926)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "classscan.txt"))
    a = ap.parse_args()

    cur = theory_sha(ROOT / "runs" / "of_s3.pt")
    files = sorted((ROOT / "runs").glob("*.pt"))
    files = [f for f in files if f.stem != "smoke"]

    env = StrategicEnduranceEnv()
    seeds = [a.seed0 + k for k in range(a.episodes)]
    opps = build_opps()

    # the bar, scored in the same field with itself removed
    me = TitForTatAgent
    sub = [(n, f) for n, f in opps if n != "TitForTat"]
    ti = [np.mean([play_episode(env, {IRAN: me(), US: fac()}, s)["u"][IRAN]
                   for s in seeds]) for _, fac in sub]
    tu = [np.mean([play_episode(env, {IRAN: fac(), US: me()}, s)["u"][US]
                   for s in seeds]) for _, fac in sub]
    bar = 0.5 * (float(np.mean(ti)) + float(np.mean(tu)))
    print("TitForTat bar: %.1f" % bar, flush=True)

    rows = []
    for f in files:
        try:
            mi, mu = seats(f, opps, seeds, env)
        except Exception as exc:
            sys.stderr.write("%s: %s\n" % (f.name, exc))
            continue
        rows.append((f.stem, mi, mu, 0.5 * (mi + mu), theory_sha(f) == cur))
        print("  %-22s I=%7.1f U=%7.1f  score=%6.1f"
              % (f.stem, mi, mu, 0.5 * (mi + mu)), flush=True)

    out = ["class-field seat scan -- %d checkpoints, %d opponents, "
           "%d episodes/seat, seed0=%d" % (len(rows), len(opps), a.episodes,
                                           a.seed0),
           "",
           "USE THIS TO RANK CANDIDATES AGAINST EACH OTHER, NOT TO READ A",
           "MARGIN OVER TitForTat. Every candidate here faces an identical",
           "opponent set, so the ranking is sound. But TitForTat's own row",
           "(%.1f) is not comparable to them: a candidate plays TitForTat"
           % bar,
           "while TitForTat, having no mirror match, does not -- and that is",
           "the matchup TitForTat is strongest in. In the real round robin it",
           "does play our submission, which lifts it by roughly 12 points.",
           "For margins, run analysis/classfield.py with a SINGLE candidate:",
           "putting several of ours in at once makes them each other's",
           "opponents, which the real tournament never does.", ""]

    out.append("best WHOLE checkpoints")
    out.append("%-4s %-22s %8s %9s %9s" % ("#", "checkpoint", "score",
                                           "Iran", "U.S."))
    for i, r in enumerate(sorted(rows, key=lambda r: -r[3])[:15], 1):
        out.append("%-4d %-22s %8.1f %9.1f %9.1f"
                   % (i, r[0], r[3], r[1], r[2]))

    graft = [r for r in rows if r[4]]
    out.append("")
    out.append("best IRAN nets (current theory only, %d of %d graftable)"
               % (len(graft), len(rows)))
    for i, r in enumerate(sorted(graft, key=lambda r: -r[1])[:8], 1):
        out.append("%-4d %-22s %9.1f  (its U.S. %.1f)" % (i, r[0], r[1], r[2]))
    out.append("")
    out.append("best U.S. nets (current theory only)")
    for i, r in enumerate(sorted(graft, key=lambda r: -r[2])[:8], 1):
        out.append("%-4d %-22s %9.1f  (its Iran %.1f)" % (i, r[0], r[2], r[1]))

    if graft:
        bi = max(graft, key=lambda r: r[1])
        bu = max(graft, key=lambda r: r[2])
        best_whole = max(rows, key=lambda r: r[3])
        out.append("")
        out.append("graft to build: --iran runs/%s.pt --us runs/%s.pt"
                   % (bi[0], bu[0]))
        out.append("  screened seats: %.1f + %.1f -> %.1f"
                   % (bi[1], bu[2], 0.5 * (bi[1] + bu[2])))
        out.append("  best whole checkpoint: %s at %.1f"
                   % (best_whole[0], best_whole[3]))
        gain = 0.5 * (bi[1] + bu[2]) - best_whole[3]
        out.append("  the graft is worth %+.1f over the best whole checkpoint "
                   "here; confirm before believing it" % gain)

    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write("\n" + text + "\n")


if __name__ == "__main__":
    main()
