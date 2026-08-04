#!/usr/bin/env python3
"""Per-opponent breakdown of a checkpoint against the reference pool.

This mirrors the "your u as Iran / your u as U.S." table SEEBoard shows for
a submission, so a candidate can be compared opponent by opponent instead of
through one aggregate.

It doubles as the calibration check for `my_team/reference_pool.py`. Three
of the six references are known exactly -- Random and TitForTat ship in the
repo, MasterAgent is the recovered benchmark checkpoint -- so if those three
columns reproduce the board's numbers, the harness (seeds, episode count,
protocol) is right and any remaining error is in the three reconstructed
agents.

Observed on the board for runs/board_reference.pt:

    Random             119.8 / 113.8      TitForTat      34.3 /  2.7
    Hawk               -12.1 / -58.7      BayesThreshold 77.6 /  9.4
    Dove               144.4 / 135.9      MasterAgent    70.1 / -2.7

    python analysis/board_detail.py <ckpt> [--episodes 40]
"""
import argparse
import io
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent              # noqa: E402
from see.config import IRAN, US                                # noqa: E402
from see.env import StrategicEnduranceEnv                      # noqa: E402
from see.tournament.runner import play_episode                 # noqa: E402
from my_team.reference_pool import reference_factories         # noqa: E402

# what the board reported for the first upload; used as a calibration target
OBSERVED = {
    "Random": (119.8, 113.8),
    "Hawk": (-12.1, -58.7),
    "Dove": (144.4, 135.9),
    "TitForTat": (34.3, 2.7),
    "BayesianThreshold": (77.6, 9.4),
    "MasterAgent": (70.1, -2.7),
}
EXACT = ("Random", "TitForTat", "MasterAgent")


def detail(ckpt, episodes, seed0):
    env = StrategicEnduranceEnv()
    seeds = [seed0 + k for k in range(episodes)]
    facs = reference_factories()
    me = CheckpointAgent(ckpt)
    rows = {}
    for name, fac in sorted(facs.items()):
        as_I = [play_episode(env, {IRAN: me, US: fac()}, s)["u"][IRAN]
                for s in seeds]
        as_U = [play_episode(env, {IRAN: fac(), US: me}, s)["u"][US]
                for s in seeds]
        rows[name] = (float(np.mean(as_I)), float(np.mean(as_U)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=20_260_901)
    ap.add_argument("--compare", action="store_true",
                    help="show the board's observed numbers alongside")
    ap.add_argument("--out", default=str(ROOT / "analysis" / "detail.txt"))
    a = ap.parse_args()

    rows = detail(a.ckpt, a.episodes, a.seed0)
    out = ["%s -- %d episodes/role, seed0=%d"
           % (pathlib.Path(a.ckpt).name, a.episodes, a.seed0), ""]
    hdr = "%-20s %9s %9s" % ("reference opponent", "as Iran", "as U.S.")
    if a.compare:
        hdr += "   %9s %9s   %s" % ("board I", "board U", "exact?")
    out.append(hdr)
    for n, (i, u) in rows.items():
        line = "%-20s %9.1f %9.1f" % (n, i, u)
        if a.compare and n in OBSERVED:
            o = OBSERVED[n]
            line += "   %9.1f %9.1f   %s" % (
                o[0], o[1], "EXACT" if n in EXACT else "fitted")
        out.append(line)

    mi = float(np.mean([v[0] for v in rows.values()]))
    mu = float(np.mean([v[1] for v in rows.values()]))
    out.append("")
    out.append("%-20s %9.1f %9.1f    score = %.1f"
               % ("MEAN", mi, mu, 0.5 * (mi + mu)))

    if a.compare:
        ex = [n for n in EXACT if n in rows]
        err = float(np.mean([abs(rows[n][k] - OBSERVED[n][k])
                             for n in ex for k in (0, 1)]))
        out.append("mean |error| on the %d exactly-known references: %.1f"
                   % (len(ex), err))

    io.open(a.out, "w", encoding="utf-8").write("\n".join(out) + "\n")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
