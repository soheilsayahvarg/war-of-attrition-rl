#!/usr/bin/env python3
"""Checkpoint selection that admits what the replica does and does not know.

The board has six slots. Four of them (Random, TitForTat, Hawk,
MasterAgent) reproduce locally to the decimal on *every one* of our four
uploads, so a score computed on those four is trustworthy. Two of them
(Dove, BayesianThreshold) do not reproduce under any parameter setting we
can find, so a score computed on our reconstruction of them is not.

Ranking on the full replica is what produced the second upload: it
predicted 70.3 where the board gave 56.8, and the whole 13.5-point error
sat in those two slots. So this script refuses to average the two kinds of
evidence together. It reports:

  TRUSTED   mean over the four exactly-reproduced references. Two thirds of
            the board's weight, with no reconstruction error in it.

  PROXY     mean over Patient and Plateau. Not a reconstruction of the two
            soft references -- their absolute levels are nothing alike --
            but analysis/proxy_probe.py shows Patient is the only local
            opponent that *orders* our uploads the way the board's soft
            slots did (spread -18.6 / -43.3 against the board's -39.5 /
            -71.3), and for the same reason: it waits, and waiting is what
            punishes an agent that pins sigma = 0.75 and never exits.

A candidate that beats the shipped checkpoint on *both* is unambiguously
better. Nothing has managed that yet -- every checkpoint since has traded
one against the other -- so in practice the EST column below decides, with
the caveats stated there.

    python analysis/select2.py runs/bp_s1.pt runs/bp_s2.pt ...
"""
import argparse
import io
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent              # noqa: E402
from see.agents.scripted import RandomAgent, TitForTatAgent    # noqa: E402
from see.config import IRAN, US                                # noqa: E402
from see.env import StrategicEnduranceEnv                      # noqa: E402
from see.tournament.runner import play_episode                 # noqa: E402
from my_team.opponents import PatientAgent, PlateauAgent       # noqa: E402
from my_team.reference_pool import HawkAgent, MASTER           # noqa: E402

TRUSTED = {
    "Random": RandomAgent,
    "TitForTat": TitForTatAgent,
    "Hawk": HawkAgent,
    "MasterAgent": lambda: CheckpointAgent(str(MASTER)),
}
PROXY = {"Patient": PatientAgent, "Plateau": PlateauAgent}

# the shipped checkpoint, for reference (measured by this same script)
SHIPPED = "runs/bh_s2.pt"   # best board score to date: 61.7

# ------------------------------------------------------------------ #
# Turning the two numbers into one, and how much to trust it
# ------------------------------------------------------------------ #
# The board score is a plain mean over six slots, so per seat
#
#     board_seat = (4 * TRUSTED_seat + 2 * SOFT_seat) / 6
#
# TRUSTED is measured exactly. SOFT (Dove and BayesianThreshold) is not
# reproducible, but each upload tells us what the real soft slots paid one
# *known* policy, so every upload buys an anchor:
#
#     upload            PROXY (I / U)      real SOFT (I / U)     board
#     board_reference   48.4 /   9.2       111.0 / 72.65          52.9
#     final_v4          34.5 / -18.1        82.9 / 39.8           56.8
#     bp_s4             34.4 /  37.2        84.25/ 68.9           58.8
#     bh_s2             34.6 /  37.5        83.15/ 80.0           61.7
#
# Caveats:
#
#   * the fit cannot be validated on its own anchors -- it passes through
#     them by construction, so a perfect match there is not evidence;
#   * outside the anchor range it does not extrapolate, it clamps, and the
#     'beyond' column reports how far outside a candidate sits;
#   * it assumes the real soft references respond to patience the way
#     Patient and Plateau do, which is the assumption proxy_probe.py
#     supports on ordering only, never on level;
#   * and, as the next block shows, the proxy has now run out of resolution
#     at the top of its range.
# Keyed on the immutable copies under runs/, never on submission.pt: that
# file is the *shipped* checkpoint and gets overwritten whenever a better
# one is found, which would silently attach a stale board score to whatever
# happened to be sitting there.
ANCHORS = {
    "runs/board_reference.pt": {"soft": (111.0, 72.65), "board": 52.9},
    "runs/final_v4.pt": {"soft": (82.9, 39.8), "board": 56.8},
    "runs/bp_s4.pt": {"soft": (84.25, 68.9), "board": 58.8},
    "runs/bh_s2.pt": {"soft": (83.15, 80.0), "board": 61.7},
    "runs/final_graft.pt": {"soft": (82.35, 38.15), "board": 58.3},
}

# ------------------------------------------------------------------ #
# THE EST COLUMN IS RETIRED. Rank on TRUSTED.
# ------------------------------------------------------------------ #
# The fifth anchor is the one that settles it. Predicted and actual, for
# the fifth upload:
#
#     slot                    predicted   actual    error
#     TRUSTED Iran (4 exact)      65.3     65.30    +0.00
#     TRUSTED U.S.  (4 exact)     49.4     49.45    +0.05
#     SOFT Iran (proxy)           83.0     82.35    -0.65
#     SOFT U.S.  (proxy)          68.9     38.15   -30.75
#
# The exactly-measured two thirds landed on the decimal. The whole 5.3-point
# error in the final score came from one interpolated number being wrong by
# thirty. The header above had already said why: at PROXY_U ~ 37 the proxy is
# saturated. This checkpoint sat at 37.1, and the real soft references scored
# its U.S. seat 38.2 where the anchors either side of it scored 68.9 and 80.0.
#
# So the proxy does not merely lose resolution up there -- it is uninformative
# there. Two checkpoints it cannot tell apart (37.1 vs 37.5) differ by 42
# points on the thing being predicted. No monotone map from PROXY_U to SOFT_U
# exists, and interpolating one produced a number that was not merely
# imprecise but qualitatively wrong: it said this checkpoint would beat the
# board leader when in fact it lost to it by 3.4.
#
# Rank candidates on TRUSTED, which is measured exactly on two thirds of the
# board's weight. Treat the soft slots as unknown rather than estimated.
#
# What DOES predict exactly: seat decomposition. A reference agent never
# plays itself, so an agent's Iran-seat games and U.S.-seat games are
# disjoint (analysis/splice.py). A spliced checkpoint therefore inherits its
# Iran parent's board Iran score and its U.S. parent's board U.S. score, with
# no interaction. That is arithmetic, not a fit -- and it is now confirmed on
# the board: the fifth upload's row is exactly og_s2_t3's Iran seat and
# oh_s2_t5's U.S. seat.
#
# The practical consequence is a hard ceiling. The measured seats we own are
#
#     board Iran:  bh_s2 72.9,  og_s2_t3 71.0
#     board U.S.:  bh_s2 50.4,  oh_s2_t5 45.7
#
# so the best board score reachable by splicing anything we have measured is
# (72.9 + 50.4) / 2 = 61.7 -- which is bh_s2 itself. Beating it needs a new
# run with a better seat, not a better combination of these.

# WHERE THIS ESTIMATE STOPS BEING USEFUL, stated plainly. The fourth anchor
# lands almost exactly on top of the third in proxy space and nowhere near
# it in outcome:
#
#     checkpoint   PROXY_U      real SOFT_U
#     bp_s4          37.2          68.9
#     bh_s2          37.5          80.0
#
# Two policies the proxy cannot tell apart (0.3 points) that the real soft
# references score 11 points apart. So above PROXY_U ~ 37 the proxy has
# saturated: it has no resolution left, and a one- or two-point difference
# in the EST column up there means nothing at all. It also explains why the
# estimate under-shot bh_s2 by 1.9 (predicted 59.8, board gave 61.7) having
# over-shot bp_s4 by 6.0 in the other direction.
#
# Practical rule from here on: rank candidates by TRUSTED, which is
# measured exactly and where all the remaining headroom now sits anyway
# (Random and MasterAgent in the U.S. seat are worth +4.2 and +2.8 board
# points respectively). Treat EST as a sanity check, not a ranking.

# The third anchor is what killed the straight line. Sorted by U.S. proxy
# score, the real soft slots paid:
#
#     PROXY_U   -18.1     9.2      37.2    37.5
#     SOFT_U     39.8    72.65     68.9    80.0
#
# It climbs steeply and then flattens, so the two-anchor linear fit
# predicted 106.2 at PROXY_U = 37.2 where the board gave 68.9 -- exactly
# the 6.0 points by which that estimate overshot, and precisely what its
# own "beyond 27.9" warning was pointing at. Hence: piecewise-linear
# interpolation between anchors, *clamped* at both ends rather than
# extrapolated.


def fit_soft_map(rows):
    """Anchor points for interpolating PROXY_seat -> SOFT_seat."""
    got = [(r, ANCHORS[r[0]]) for r in rows if r[0] in ANCHORS]
    if len(got) < 2:
        return None
    coef = []
    for k in (0, 1):                                    # 0 = Iran, 1 = U.S.
        pts = sorted((r[5][k], a["soft"][k]) for r, a in got)
        coef.append((np.array([p[0] for p in pts], dtype=float),
                     np.array([p[1] for p in pts], dtype=float)))
    return coef


def estimate(coef, trusted_seats, proxy_seats):
    """Predicted board score, plus how far outside the anchors we are."""
    out, far = [], 0.0
    for k in (0, 1):
        x, y = coef[k]
        soft = float(np.interp(proxy_seats[k], x, y))   # np.interp clamps
        out.append((4.0 * trusted_seats[k] + 2.0 * soft) / 6.0)
        far = max(far, max(0.0, proxy_seats[k] - x[-1], x[0] - proxy_seats[k]))
    return 0.5 * (out[0] + out[1]), out[0], out[1], far


def group(me, pool, seeds, env):
    per = {}
    for nm, fac in pool.items():
        i = np.mean([play_episode(env, {IRAN: me, US: fac()}, s)["u"][IRAN]
                     for s in seeds])
        u = np.mean([play_episode(env, {IRAN: fac(), US: me}, s)["u"][US]
                     for s in seeds])
        per[nm] = (float(i), float(u))
    mi = float(np.mean([v[0] for v in per.values()]))
    mu = float(np.mean([v[1] for v in per.values()]))
    return per, mi, mu, 0.5 * (mi + mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=20_260_901)
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "analysis" / "select2.txt"))
    a = ap.parse_args()

    env = StrategicEnduranceEnv()
    seeds = [a.seed0 + k for k in range(a.episodes)]
    todo = list(a.ckpts)
    for anchor in reversed(list(ANCHORS)):        # both calibration points
        if anchor not in todo:
            todo.insert(0, anchor)

    out = ["selection on trusted slots and a behavioural proxy",
           "(%d episodes/role, seed0=%d)" % (a.episodes, a.seed0), "",
           "%-18s %19s %19s" % ("checkpoint",
                                "TRUSTED (4 exact)", "PROXY (patient)"),
           "%-18s %9s %9s %9s %9s" % ("", "Iran", "U.S.", "Iran", "U.S.")]
    rows = []
    for c in todo:
        me = CheckpointAgent(str(ROOT / c))
        tp, ti, tu, ts = group(me, TRUSTED, seeds, env)
        pp, pi, pu, ps = group(me, PROXY, seeds, env)
        rows.append((c, ts, ps, tp, pp, (pi, pu), (ti, tu)))
        out.append("%-18s %9.1f %9.1f %9.1f %9.1f   T=%6.1f  P=%6.1f"
                   % (pathlib.Path(c).stem, ti, tu, pi, pu, ts, ps))

    base = next(r for r in rows if r[0] == SHIPPED)
    out.append("")
    out.append("Pareto test against %s (T=%.1f, P=%.1f):"
               % (SHIPPED, base[1], base[2]))
    win = [r for r in rows if r[0] != SHIPPED
           and r[1] > base[1] and r[2] > base[2]]
    if win:
        for r in sorted(win, key=lambda r: -(r[1] + r[2])):
            out.append("  BETTER ON BOTH  %-18s T=%+6.1f  P=%+6.1f"
                       % (pathlib.Path(r[0]).stem, r[1] - base[1],
                          r[2] - base[2]))
    else:
        out.append("  none improves both -- fall back to the estimate below,"
                   " and read its caveats in the header.")

    coef = fit_soft_map(rows)
    if coef is not None:
        out.append("")
        out.append("estimated board score  (two-anchor interpolation --"
                   " see header; 'beyond' is")
        out.append("how far a candidate's PROXY sits outside the two"
                   " anchors, in points)")
        out.append("  %-18s %9s %9s %9s %9s"
                   % ("checkpoint", "est Iran", "est U.S.", "EST", "beyond"))
        est = []
        for r in rows:
            e, ei, eu, far = estimate(coef, r[6], r[5])
            est.append((e, r[0], far))
            mark = ""
            if r[0] in ANCHORS:
                mark = "   <- anchor, board said %.1f" % ANCHORS[r[0]]["board"]
            out.append("  %-18s %9.1f %9.1f %9.1f %9.1f%s"
                       % (pathlib.Path(r[0]).stem, ei, eu, e, far, mark))
        best = max(e for e in est if e[1] not in ANCHORS) \
            if any(e[1] not in ANCHORS for e in est) else None
        if best is not None:
            out.append("")
            out.append("  best candidate: %s  est %.1f (%+.1f vs the shipped"
                       " %.1f board score)"
                       % (pathlib.Path(best[1]).stem, best[0],
                          best[0] - ANCHORS[SHIPPED]["board"],
                          ANCHORS[SHIPPED]["board"]))
            if best[2] > 5.0:
                out.append("  WARNING: that candidate is %.1f points outside"
                           " the anchor range on PROXY," % best[2])
                out.append("  so its estimate is an extrapolation, not an"
                           " interpolation. Treat as a direction,")
                out.append("  not a number. Uploading is still free -- the"
                           " board keeps the best scored upload.")

    if a.detail:
        out.append("")
        for c, ts, ps, tp, pp in rows:
            out.append(pathlib.Path(c).stem)
            for nm, (i, u) in list(tp.items()) + list(pp.items()):
                out.append("    %-16s %9.1f %9.1f" % (nm, i, u))

    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
