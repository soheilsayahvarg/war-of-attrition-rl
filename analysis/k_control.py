#!/usr/bin/env python3
"""Proposition 4 with the obvious confounder held fixed.

P4 predicts P(voluntary exit) *falls* as the player's own accumulated
public commitment K rises, because X_i = x0 - lam*K - ... So the raw
cross-tab should show a negative slope.

For the submitted checkpoint the raw cross-tab shows the *opposite* sign.
The reason is that K is not exogenous: it is a discounted sum of the
player's own past signals, so it can only be large at late stages, and
late stages are exactly where endurance has been spent and exit (forced or
voluntary) becomes likely. Stage index and remaining endurance are both
confounders, and both push the raw slope the wrong way.

This script re-runs the same cross-tab inside bins of stage index and of
remaining endurance, which is the standard fix.

    python analysis/k_control.py <ckpt> [--episodes 240]
"""
import argparse
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from propositions import collect                              # noqa: E402


def rate(rs):
    legal = [r for r in rs if r["could_exit"]]
    if not legal:
        return None, 0
    return sum(r["exit"] for r in legal) / len(legal), len(legal)


def cross(rows, out):
    # Split at the median K of this policy rather than a fixed constant.
    # A fixed cut only works if the policy actually spreads over it; the
    # submitted agent holds a sigma = 0.75 posture most of the time, so its
    # K is above any fixed low threshold from stage 1 onward and a constant
    # cut would put 95% of the stages in one bin.
    import statistics
    cut = statistics.median([r["K"] for r in rows]) if rows else 0.5
    out.append("median K = %.2f (used as the low/high cut)" % cut)
    out.append("")
    hi = [r for r in rows if r["K"] > cut]
    lo = [r for r in rows if r["K"] <= cut]
    pl, nl = rate(lo)
    ph, nh = rate(hi)
    out.append("raw (no controls)")
    out.append("  K low  P(exit) = %.4f  (n=%d)" % (pl, nl))
    out.append("  K high P(exit) = %.4f  (n=%d)" % (ph, nh))
    out.append("")

    out.append("holding the stage index fixed")
    out.append("  %-12s %10s %8s %10s %8s" % ("stage band", "P|K low", "n",
                                              "P|K high", "n"))
    for lohi in [(0, 4), (5, 9), (10, 14), (15, 40)]:
        band = [r for r in rows if lohi[0] <= r["t"] <= lohi[1]]
        pl, nl = rate([r for r in band if r["K"] <= cut])
        ph, nh = rate([r for r in band if r["K"] > cut])
        out.append("  t=%-10s %10s %8d %10s %8d" % (
            "%d..%d" % lohi,
            "%.4f" % pl if pl is not None else "--", nl,
            "%.4f" % ph if ph is not None else "--", nh))
    out.append("")

    out.append("holding remaining endurance fixed")
    out.append("  %-12s %10s %8s %10s %8s" % ("e_hat band", "P|K low", "n",
                                              "P|K high", "n"))
    for lohi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 2.0)]:
        band = [r for r in rows if lohi[0] <= r["e_hat"] < lohi[1]]
        pl, nl = rate([r for r in band if r["K"] <= cut])
        ph, nh = rate([r for r in band if r["K"] > cut])
        out.append("  e=%-10s %10s %8d %10s %8d" % (
            "%.2f-%.2f" % lohi,
            "%.4f" % pl if pl is not None else "--", nl,
            "%.4f" % ph if ph is not None else "--", nh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=240)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows, _ = collect(a.ckpt, a.episodes)
    rows = [r for r in rows if not r["forced"]]
    out = ["proposition 4 with controls -- %s (%d episodes per pairing)"
           % (pathlib.Path(a.ckpt).name, a.episodes), ""]
    cross(rows, out)
    text = "\n".join(out)
    path = a.out or str(ROOT / "analysis" / "k_control.txt")
    io.open(path, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write("wrote " + path + "\n")


if __name__ == "__main__":
    main()
