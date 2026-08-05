#!/usr/bin/env python3
"""What, behaviourally, separates two checkpoints in front of a passive foe.

The board's Dove row discriminates sharply between our two uploads
(144.4 / 135.9 then 104.9 / 64.6) while every fixed-threshold Dove we can
write gives them the same score. So the real Dove is reacting to something
our two policies do *differently*. This dumps the obvious candidates --
signal level, stage count, who leaves first -- against a fixed passive
opponent, so the hypothesis can be chosen from data instead of guessed.

    python analysis/sigprofile.py runs/board_reference.pt submission.pt
"""
import argparse
import io
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.base import Agent                              # noqa: E402
from see.agents.checkpoint import CheckpointAgent              # noqa: E402
from see.config import (ACTION_EXIT, ACTION_HOLD, IRAN,        # noqa: E402
                        SIGNAL_LEVELS, US)
from see.env import StrategicEnduranceEnv                      # noqa: E402


class Passive(Agent):
    """Signals zero, never exits voluntarily. A pure measuring stick."""
    name = "Passive"

    def act(self, obs, legal_mask, public_state=None):
        a = int(np.argmin([abs(s) for s in SIGNAL_LEVELS]))
        return a if legal_mask[a] else self.first_legal(legal_mask)


def profile(ckpt, seeds, env):
    me = CheckpointAgent(ckpt)
    sig, ln, quit_first, first_sig, hold = [], [], [], [], []
    for seat, other in ((IRAN, US), (US, IRAN)):
        for s in seeds:
            agents = {seat: me, other: Passive()}
            # replay the episode exactly as see/tournament/runner.py does,
            # but keep our own action stream so it can be summarised
            obs, _ = env.reset(seed=s)
            for pid, ag in agents.items():
                ag.reset(pid, seed=s * 7919 + (1 if pid == IRAN else 2))
            done, mine = False, []
            while not done:
                public = env.public_state()
                acts = {}
                for pid, ag in agents.items():
                    mask = env.get_legal_mask(pid)
                    a = int(ag.act(obs[pid], mask, public))
                    if not mask[a]:
                        a = int(np.argmax(mask))
                    acts[pid] = a
                mine.append(acts[seat])
                obs, _r, term, trunc, _info = env.step(acts)
                done = term[IRAN] or trunc[IRAN]
            t = env.t
            lv = [SIGNAL_LEVELS[a] for a in mine if a < len(SIGNAL_LEVELS)]
            if lv:
                sig.append(float(np.mean(lv)))
                first_sig.append(float(lv[0]))
            hold.append(float(np.mean([a == ACTION_HOLD for a in mine])))
            ln.append(t)
            quit_first.append(1.0 if mine[-1] == ACTION_EXIT else 0.0)
    return {
        "mean signal": float(np.mean(sig)),
        "max signal": float(np.max(sig)),
        "first-stage signal": float(np.mean(first_sig)),
        "frac episodes mean sigma=1": float(np.mean([s >= 0.999 for s in sig])),
        "frac stages HOLD": float(np.mean(hold)),
        "episode length": float(np.mean(ln)),
        "we exit first": float(np.mean(quit_first)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=20_260_901)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "sigprofile.txt"))
    a = ap.parse_args()

    env = StrategicEnduranceEnv()
    seeds = [a.seed0 + k for k in range(a.episodes)]
    res = {c: profile(str(ROOT / c), seeds, env) for c in a.ckpts}
    keys = list(next(iter(res.values())).keys())
    out = ["behaviour against a purely passive opponent", ""]
    out.append("%-24s" % "statistic"
               + "".join(" %20s" % pathlib.Path(c).stem for c in a.ckpts))
    for k in keys:
        out.append("%-24s" % k
                   + "".join(" %20.3f" % res[c][k] for c in a.ckpts))
    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
