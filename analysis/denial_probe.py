#!/usr/bin/env python3
"""Is conceding to an identified commitment type the right tournament move?

Against an opponent that never concedes, quitting early is the per-episode
best response (report, finding 4).  But the leaderboard criterion is
*relative*: your score minus the reference agent's.  Quitting hands the whole
prize B to the opponent, so the same act that maximises your own utility also
maximises theirs.

This probe measures the trade exactly.  It plays the submitted checkpoint,
except that in the U.S. seat it stops conceding once the opponent has been
identified as an exact mirror (the defining signature of Tit-for-Tat: its
signal at t is our signal at t-1).  Detection is behavioural and needs no
knowledge of who the opponent is.

    python analysis/denial_probe.py [--episodes 60]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.base import Agent                                  # noqa: E402
from see.agents.checkpoint import CheckpointAgent                  # noqa: E402
from see.config import (ACTION_EXIT, ACTION_HOLD, IRAN,            # noqa: E402
                        SIGNAL_LEVELS, US)
from see.tournament.runner import run_tournament                   # noqa: E402

SIG_OWN, SIG_OPP = 6, 7


class ConditionalDenier(Agent):
    """Checkpoint policy + a commitment-type override on the U.S. seat."""

    def __init__(self, path, hold_sigma=0.75, n_match=3, name="ME+deny"):
        self.inner = CheckpointAgent(path, sample=True)
        self.hold_sigma = hold_sigma
        self.n_match = n_match
        self.name = name

    def reset(self, player_id, seed=None):
        self.pid = player_id
        self.inner.reset(player_id, seed)
        self.prev_own_sig = None
        self.matches = 0
        self.checks = 0

    def _update_detector(self, obs):
        """Did the opponent just replay our previous signal?"""
        if self.prev_own_sig is not None:
            self.checks += 1
            if abs(float(obs[SIG_OPP]) - self.prev_own_sig) < 1e-6:
                self.matches += 1
            else:
                self.matches = 0
        self.prev_own_sig = float(obs[SIG_OWN])

    def act(self, obs, legal_mask, public_state=None):
        self._update_detector(obs)
        a = self.inner.act(obs, legal_mask, public_state)
        identified = (self.pid == US and self.matches >= self.n_match)
        if identified and a == ACTION_EXIT:
            want = int(np.argmin([abs(self.hold_sigma - s)
                                  for s in SIGNAL_LEVELS]))
            for cand in (want, ACTION_HOLD, want - 1, want + 1):
                if 0 <= cand < len(legal_mask) and legal_mask[cand] \
                        and cand != ACTION_EXIT:
                    return cand
            return int(np.argmax(legal_mask))
        return a


def show(tag, entries, episodes, seed0, me_key):
    r = run_tournament(entries, episodes_per_side=episodes, seed0=seed0,
                       verbose=False)
    b = {row["name"]: row for row in r["leaderboard"]}
    mine = b[me_key]
    tft = next(v for k, v in b.items() if "TitForTat" in k)
    edge = mine["score"] - tft["score"]
    print("%-26s pool=%d  ME=%6.1f (I %6.1f / U %6.1f)   TFT=%6.1f   edge=%+7.1f"
          % (tag, len(b), mine["score"], mine["score_as_iran"],
             mine["score_as_us"], tft["score"], edge), flush=True)
    return {"tag": tag, "pool": len(b), "me": mine["score"],
            "me_I": mine["score_as_iran"], "me_U": mine["score_as_us"],
            "tft": tft["score"], "edge": edge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--me", default=str(ROOT / "submission.pt"))
    a = ap.parse_args()

    plain = CheckpointAgent(a.me)
    plain.name = "ME"

    field = []
    for nm, p in [("master", ROOT / "benchmarks" / "master_agent.pt"),
                  ("refspec_s3", ROOT / "runs" / "refspec_s3.pt"),
                  ("rich_s0", ROOT / "runs" / "rich_s0.pt"),
                  ("rich_s1", ROOT / "runs" / "rich_s1.pt"),
                  ("richshape_s0", ROOT / "runs" / "richshape_s0.pt")]:
        if p.exists():
            field.append((nm, CheckpointAgent(str(p))))

    rows = []
    print("\n--- reference evaluation (the scripts/evaluate.py pool) ---")
    rows.append(show("plain checkpoint", {"ME": plain}, a.episodes, a.seed0, "ME"))
    for hs in (0.0, 0.25, 0.75):
        d = ConditionalDenier(a.me, hold_sigma=hs, name="ME+deny")
        rows.append(show("deny, hold sigma=%.2f" % hs, {"ME+deny": d},
                         a.episodes, a.seed0, "ME+deny"))

    print("\n--- class-sized pool (%d extra entrants) ---" % len(field))
    ent = {"ME": plain}
    ent.update(dict(field))
    rows.append(show("plain checkpoint", ent, a.episodes, a.seed0, "ME"))
    for hs in (0.0, 0.25, 0.75):
        d = ConditionalDenier(a.me, hold_sigma=hs, name="ME+deny")
        ent = {"ME+deny": d}
        ent.update(dict(field))
        rows.append(show("deny, hold sigma=%.2f" % hs, ent,
                         a.episodes, a.seed0, "ME+deny"))

    (ROOT / "analysis" / "denial_probe.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
