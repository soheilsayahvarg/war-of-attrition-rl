#!/usr/bin/env python3
"""Fit the two reconstructed references that are still wrong.

`analysis/board_detail.py` reproduces the board's per-opponent numbers for
our first upload exactly against Random, TitForTat, MasterAgent (all three
known exactly) and -- as it turned out -- against Hawk too, which confirms
"always escalate, never concede" is the right reading.

Dove and BayesianThreshold are still off, and there the board gives four
exact target numbers for a *fixed* checkpoint:

    Dove              us as Iran 144.4    us as U.S. 135.9
    BayesianThreshold us as Iran  77.6    us as U.S.   9.4

That is a small, well-posed fit: hold the checkpoint fixed, sweep the
parameters of the two agents, minimise the error on those four numbers.

    python analysis/fit_refs.py runs/board_reference.pt
"""
import argparse
import io
import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent              # noqa: E402
from see.config import IRAN, US                                # noqa: E402
from see.env import StrategicEnduranceEnv                      # noqa: E402
from see.tournament.runner import play_episode                 # noqa: E402
from my_team.reference_pool import (BayesianThresholdAgent,    # noqa: E402
                                    DoveAgent)

TARGET_DOVE = (144.4, 135.9)
TARGET_BT = (77.6, 9.4)


def score(me, factory, seeds, env):
    as_I = [play_episode(env, {IRAN: me, US: factory()}, s)["u"][IRAN]
            for s in seeds]
    as_U = [play_episode(env, {IRAN: factory(), US: me}, s)["u"][US]
            for s in seeds]
    return float(np.mean(as_I)), float(np.mean(as_U))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=20_260_901)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "fit_refs.txt"))
    a = ap.parse_args()

    env = StrategicEnduranceEnv()
    seeds = [a.seed0 + k for k in range(a.episodes)]
    me = CheckpointAgent(a.ckpt)
    out = []

    out.append("Dove: sweep the endurance threshold theta")
    out.append("  %-8s %9s %9s   %s" % ("theta", "as Iran", "as U.S.", "err"))
    best_d = None
    for th in (1.00, 0.95, 0.90, 0.85, 0.80, 0.70):
        i, u = score(me, lambda th=th: DoveAgent(theta=th), seeds, env)
        e = abs(i - TARGET_DOVE[0]) + abs(u - TARGET_DOVE[1])
        out.append("  %-8.2f %9.1f %9.1f   %6.1f" % (th, i, u, e))
        if best_d is None or e < best_d[0]:
            best_d = (e, th, i, u)
    out.append("  target   %9.1f %9.1f" % TARGET_DOVE)
    out.append("  best theta = %.2f (err %.1f)" % (best_d[1], best_d[0]))
    out.append("")

    out.append("BayesianThreshold: sweep fold / push_below / e_fold")
    out.append("  %-6s %-6s %-6s %9s %9s   %s"
               % ("fold", "push<", "e_fold", "as Iran", "as U.S.", "err"))
    best_b = None
    grid = itertools.product((0.65, 0.85, 1.10), (0.375, 0.70),
                             (2.0, 0.35))
    for fold, pb, ef in grid:
        def fac(fold=fold, pb=pb, ef=ef):
            ag = BayesianThresholdAgent(fold=fold, push_below=pb)
            ag.e_fold = ef
            return ag
        i, u = score(me, fac, seeds, env)
        e = abs(i - TARGET_BT[0]) + abs(u - TARGET_BT[1])
        out.append("  %-6.2f %-6.2f %-6.2f %9.1f %9.1f   %6.1f"
                   % (fold, pb, ef, i, u, e))
        if best_b is None or e < best_b[0]:
            best_b = (e, (fold, pb, ef), i, u)
    out.append("  target %19s %9.1f %9.1f" % ("", TARGET_BT[0], TARGET_BT[1]))
    out.append("  best fold=%.2f push_below=%.2f e_fold=%.2f (err %.1f)"
               % (best_b[1][0], best_b[1][1], best_b[1][2], best_b[0]))

    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
