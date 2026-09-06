#!/usr/bin/env python3
"""Per-seat frontier of the official `scripts/evaluate.py` criterion.

In the reference evaluation the pool is {you, Random, TitForTat} and an agent
never plays itself, so your Iran net and your U.S. net face *disjoint* sets of
games.  Writing

    A = u_I(you as Iran   vs Random as U.S.)
    B = u_I(you as Iran   vs TFT    as U.S.)   b' = u_U(TFT as U.S. vs you)
    C = u_U(you as U.S.   vs Random as Iran)
    D = u_U(you as U.S.   vs TFT    as Iran)   d' = u_I(TFT as Iran vs you)
    R1 = u_I(TFT as Iran vs Random), R2 = u_U(TFT as U.S. vs Random)

the two leaderboard scores are

    score(you) = (A + B + C + D) / 4
    score(TFT) = (R1 + R2 + b' + d') / 4

so the headline criterion score(you) > score(TFT) is exactly

    J_I + J_U  >  R1 + R2      with   J_I = A + B - b',  J_U = C + D - d'.

R1 and R2 do not depend on us at all.  J_I depends only on the Iran seat and
J_U only on the U.S. seat, so the two seats can be optimised independently --
which is what this script measures.

    python analysis/frontier.py [--episodes 60] [--checkpoint submission.pt]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.base import Agent                                  # noqa: E402
from see.agents.scripted import RandomAgent, TitForTatAgent        # noqa: E402
from see.config import ACTION_EXIT, IRAN, SIGNAL_LEVELS, US        # noqa: E402
from see.env import StrategicEnduranceEnv                          # noqa: E402
from see.tournament.runner import play_episode                     # noqa: E402

E_HAT, K_OWN, SIG_OPP, MED = 3, 4, 7, 8


# --------------------------------------------------------------------- #
# a small parametric family of postures
# --------------------------------------------------------------------- #
class Posture(Agent):
    """sigma level + exit rule.

    exit rules
      never      : never exits voluntarily (leaves only when forced out)
      ehat<thr   : classic endurance stop-loss
      med        : exit the first time the mediation window is open and
                   t >= t_min  (the "constructed exit" of the report)
    """

    def __init__(self, sigma, rule="never", thr=0.0, t_min=0, cap=None):
        self.sigma, self.rule, self.thr, self.t_min, self.cap = \
            sigma, rule, thr, t_min, cap
        tag = {"never": "never", "ehat": f"e<{thr}", "med": f"med>={t_min}"}[rule]
        self.name = f"s{sigma}/{tag}" + (f"/cap{cap}" if cap is not None else "")
        self.t = 0

    def reset(self, player_id, seed=None):
        super().reset(player_id, seed)
        self.t = 0

    def act(self, obs, legal_mask, public_state=None):
        t, self.t = self.t, self.t + 1
        if legal_mask[ACTION_EXIT]:
            if self.rule == "ehat" and obs[E_HAT] < self.thr:
                return ACTION_EXIT
            if self.rule == "med" and obs[MED] > 0.5 and t >= self.t_min:
                return ACTION_EXIT
        s = self.sigma
        if self.cap is not None:
            s = min(self.cap, max(self.sigma, obs[SIG_OPP]))
        a = int(np.argmin([abs(s - x) for x in SIGNAL_LEVELS]))
        return a if legal_mask[a] else self.first_legal(legal_mask)


class CheckpointSeat(Agent):
    """Wraps a CheckpointAgent so it can be dropped into either seat."""

    def __init__(self, path, sample=True):
        from see.agents.checkpoint import CheckpointAgent
        self.inner = CheckpointAgent(path, sample=sample)
        self.name = "checkpoint"

    def reset(self, player_id, seed=None):
        self.inner.reset(player_id, seed)

    def act(self, obs, legal_mask, public_state=None):
        return self.inner.act(obs, legal_mask, public_state)


# --------------------------------------------------------------------- #
def duel(env, iran, us, seeds):
    """Mean (u_I, u_U) over the seed list."""
    rows = [play_episode(env, {IRAN: iran, US: us}, s) for s in seeds]
    return (float(np.mean([r["u"][IRAN] for r in rows])),
            float(np.mean([r["u"][US] for r in rows])),
            float(np.mean([r["len"] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--checkpoint", default=str(ROOT / "submission.pt"))
    ap.add_argument("--out", default=str(ROOT / "analysis" / "frontier.json"))
    a = ap.parse_args()

    seeds = [a.seed0 + k for k in range(a.episodes)]
    env = StrategicEnduranceEnv()

    # ---- the two constants of the criterion --------------------------- #
    R1, _, _ = duel(env, TitForTatAgent(), RandomAgent(), seeds)
    _, R2, _ = duel(env, RandomAgent(), TitForTatAgent(), seeds)
    target = R1 + R2
    print(f"R1 (TFT-Iran vs Random) = {R1:7.1f}")
    print(f"R2 (TFT-U.S. vs Random) = {R2:7.1f}")
    print(f"=> need  J_I + J_U > {target:.1f}\n")

    # ---- candidate postures ------------------------------------------ #
    cands = [CheckpointSeat(a.checkpoint)]
    for sig in (0.0, 0.25, 0.5, 0.75, 1.0):
        cands.append(Posture(sig, "never"))
        for thr in (0.10, 0.20, 0.35):
            cands.append(Posture(sig, "ehat", thr=thr))
        for tm in (0, 3, 6):
            cands.append(Posture(sig, "med", t_min=tm))
    for cap in (0.5, 0.75, 1.0):
        cands.append(Posture(0.0, "never", cap=cap))
        cands.append(Posture(0.0, "med", t_min=3, cap=cap))

    iran_rows, us_rows = [], []
    hdr_i = "%-22s %7s %7s %7s %8s" % ("posture", "A", "B", "b'", "J_I")
    hdr_u = "%-22s %7s %7s %7s %8s" % ("posture", "C", "D", "d'", "J_U")
    print(hdr_i)
    for c in cands:
        A, _, _ = duel(env, c, RandomAgent(), seeds)
        B, bp, ln = duel(env, c, TitForTatAgent(), seeds)
        J = A + B - bp
        iran_rows.append({"name": c.name, "A": A, "B": B, "bp": bp,
                          "J": J, "len": ln})
        print(f"{c.name:22s} {A:7.1f} {B:7.1f} {bp:7.1f} {J:8.1f}", flush=True)

    print("\n" + hdr_u)
    for c in cands:
        _, C, _ = duel(env, RandomAgent(), c, seeds)
        dp, D, ln = duel(env, TitForTatAgent(), c, seeds)
        J = C + D - dp
        us_rows.append({"name": c.name, "C": C, "D": D, "dp": dp,
                        "J": J, "len": ln})
        print(f"{c.name:22s} {C:7.1f} {D:7.1f} {dp:7.1f} {J:8.1f}", flush=True)

    iran_rows.sort(key=lambda r: -r["J"])
    us_rows.sort(key=lambda r: -r["J"])
    bi, bu = iran_rows[0], us_rows[0]
    own = (bi["A"] + bi["B"] + bu["C"] + bu["D"]) / 4
    tft = (R1 + R2 + bi["bp"] + bu["dp"]) / 4
    print("\n" + "=" * 70)
    print(f"selected Iran seat : {bi['name']:22s} J_I = {bi['J']:8.1f}")
    print(f"selected U.S. seat : {bu['name']:22s} J_U = {bu['J']:8.1f}")
    print(f"J_I + J_U = {bi['J'] + bu['J']:.1f}   (need > {target:.1f})")
    print(f"=> you {own:.1f}   TitForTat {tft:.1f}   edge {own - tft:+.1f}")

    pathlib.Path(a.out).write_text(json.dumps(
        {"R1": R1, "R2": R2, "target": target,
         "iran": iran_rows, "us": us_rows}, indent=1))


if __name__ == "__main__":
    main()
