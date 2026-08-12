#!/usr/bin/env python3
"""Score every checkpoint's two seats separately, to pick a graft.

analysis/graft.py showed the two role networks are fully separable: pairing
og_s2's Iran net with of_s3's U.S. net produced exactly og_s2's Iran score
and exactly of_s3's U.S. score, and the margin over TitForTat went from
+1.3 to +8.5. So the best submission is not the best checkpoint -- it is the
best Iran net and the best U.S. net, which need not come from the same run.

That turns selection into two independent argmaxes over the whole run
directory instead of one argmax over combined scores, and a combined score
actively hides the answer: a run whose Iran net is the best we have can sit
in the middle of the table because its U.S. net is poor, and every ranking
so far would have discarded it.

Only checkpoints whose embedded theory.py matches the current one are
scanned. graft.py refuses to mix theory sources -- the embedded source is
what recomputes the extra features at evaluation time, and there is room for
only one copy -- so the others cannot contribute a seat anyway.

This is a screen, not a verdict: fewer episodes and a smaller opponent set
than analysis/head2head.py, chosen to rank 70-odd checkpoints in reasonable
time. Feed its winners back through head2head.py at full resolution.

    python analysis/seatscan.py --episodes 20
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

# a screening subset of the head2head opponent set: the two graded scripted
# references, the instructor benchmark, and two learned agents that sit at
# opposite ends of the behavioural range (final_v4 never exits, bq_s1 does)
SCREEN_OPPS = [
    ("Random", None), ("TitForTat", None),
    ("master", "benchmarks/master_agent.pt"),
    ("final_v4", "runs/final_v4.pt"),
    ("bq_s1", "runs/bq_s1.pt"),
]


def theory_sha(path):
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return None
    src = ck.get("theory_source") or ""
    return hashlib.sha256(src.encode()).hexdigest()


def build_opps():
    f = []
    for name, rel in SCREEN_OPPS:
        if rel is None:
            f.append((name, RandomAgent if name == "Random" else TitForTatAgent))
        else:
            p = ROOT / rel
            if p.exists():
                f.append((name, lambda q=str(p): CheckpointAgent(q)))
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
    ap.add_argument("--seed0", type=int, default=98_765_431)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "seatscan.txt"))
    a = ap.parse_args()

    want = theory_sha(ROOT / "runs" / "of_s3.pt")
    files = sorted((ROOT / "runs").glob("*.pt"))
    files = [f for f in files if "_t" not in f.stem and
             not f.stem.startswith("gr_") and f.stem != "smoke"]

    env = StrategicEnduranceEnv()
    seeds = [a.seed0 + k for k in range(a.episodes)]
    opps = build_opps()

    rows, skipped = [], 0
    for f in files:
        if theory_sha(f) != want:
            skipped += 1
            continue
        try:
            mi, mu = seats(f, opps, seeds, env)
        except Exception as exc:
            sys.stderr.write("%s: %s\n" % (f.name, exc))
            continue
        rows.append((f.stem, mi, mu))
        print("  %-18s I=%7.1f  U=%7.1f" % (f.stem, mi, mu), flush=True)

    out = ["seat scan -- %d checkpoints on the current theory (%d skipped "
           "for a different one)" % (len(rows), skipped),
           "%d opponents, %d episodes/seat, seed0=%d -- a SCREEN, not a "
           "verdict" % (len(opps), a.episodes, a.seed0), ""]

    out.append("best IRAN nets")
    out.append("%-4s %-18s %9s %9s" % ("#", "checkpoint", "Iran", "(U.S.)"))
    for i, (n, mi, mu) in enumerate(sorted(rows, key=lambda r: -r[1])[:12], 1):
        out.append("%-4d %-18s %9.1f %9.1f" % (i, n, mi, mu))

    out.append("")
    out.append("best U.S. nets")
    out.append("%-4s %-18s %9s %9s" % ("#", "checkpoint", "U.S.", "(Iran)"))
    for i, (n, mi, mu) in enumerate(sorted(rows, key=lambda r: -r[2])[:12], 1):
        out.append("%-4d %-18s %9.1f %9.1f" % (i, n, mu, mi))

    if rows:
        bi = max(rows, key=lambda r: r[1])
        bu = max(rows, key=lambda r: r[2])
        best_single = max(rows, key=lambda r: 0.5 * (r[1] + r[2]))
        out.append("")
        out.append("graft to build:  --iran runs/%s.pt  --us runs/%s.pt"
                   % (bi[0], bu[0]))
        out.append("  screened seats: Iran %.1f + U.S. %.1f -> %.1f"
                   % (bi[1], bu[2], 0.5 * (bi[1] + bu[2])))
        out.append("  best single checkpoint here: %s at %.1f"
                   % (best_single[0],
                      0.5 * (best_single[1] + best_single[2])))
        out.append("  (screen only -- confirm with analysis/head2head.py)")

    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write("\n" + text + "\n")


if __name__ == "__main__":
    main()
