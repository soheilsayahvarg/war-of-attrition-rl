#!/usr/bin/env python3
"""Reproduce the SEEBoard practice leaderboard locally.

The published board scores every agent against the *reference pool*: five
scripted baselines plus the instructor benchmark, both roles, 40 episodes
per role, practice seed base 20260901. Our repo only ships two of those
five baselines (Random, TitForTat); DESIGN.md Sec. 11 describes the other
three -- an always-escalate Hawk, an always-fold Dove and a
Bayesian-threshold agent -- as "now-removed" and says reintroducing them
makes the trainer and tournament pick them up automatically.

Two things have to be pinned down to reproduce the board: what those three
agents do, and whether "against the reference pool" includes an agent's own
mirror match. Both are settled by fitting the six published rows:

    MasterAgent         56.3   I 65.7   U 46.9
    TitForTat           53.8   I 69.7   U 37.9
    Hawk                51.5   I 71.9   U 31.2
    Random              24.7   I 30.7   U 18.6
    Dove                20.3   I 25.7   U 14.9
    BayesianThreshold   19.5   I 33.7   U  5.2

Nothing under see/ is modified.

    python analysis/board.py [--me submission.pt] [--self-play] [--variant V]
"""
import argparse
import io
import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.base import Agent                              # noqa: E402
from see.agents.checkpoint import CheckpointAgent              # noqa: E402
from see.agents.scripted import RandomAgent, TitForTatAgent    # noqa: E402
from see.config import ACTION_EXIT, IRAN, SIGNAL_LEVELS, US    # noqa: E402
from see.env import StrategicEnduranceEnv                      # noqa: E402
from see.tournament.runner import play_episode                 # noqa: E402

T_FRAC, RHO, M_HAT, E_HAT = 0, 1, 2, 3
K_OWN, K_OPP, SIG_OWN, SIG_OPP, MED = 4, 5, 6, 7, 8

PUBLISHED = {
    "MasterAgent": (56.3, 65.7, 46.9),
    "TitForTat": (53.8, 69.7, 37.9),
    "Hawk": (51.5, 71.9, 31.2),
    "Random": (24.7, 30.7, 18.6),
    "Dove": (20.3, 25.7, 14.9),
    "BayesianThreshold": (19.5, 33.7, 5.2),
}


def _sig(level):
    return int(np.argmin([abs(level - s) for s in SIGNAL_LEVELS]))


class Hawk(Agent):
    """Always escalate; concede only when forced."""
    name = "Hawk"

    def act(self, obs, legal_mask, public_state=None):
        a = _sig(1.0)
        return a if legal_mask[a] else self.first_legal(legal_mask)


class DoveFold(Agent):
    """Always fold: take the exit as soon as it is legal."""
    name = "Dove"

    def act(self, obs, legal_mask, public_state=None):
        if legal_mask[ACTION_EXIT]:
            return ACTION_EXIT
        a = _sig(0.0)
        return a if legal_mask[a] else self.first_legal(legal_mask)


class DoveCalm(Agent):
    """Never escalates, never concedes voluntarily."""
    name = "Dove"

    def act(self, obs, legal_mask, public_state=None):
        a = _sig(0.0)
        return a if legal_mask[a] else self.first_legal(legal_mask)


class BayesianThreshold(Agent):
    """Folds cheaply against a committed aggressor, pressures weak types."""
    name = "BayesianThreshold"

    def __init__(self, fold=0.5, push=0.75, decay=0.7, push_below=0.125,
                 e_fold=2.0):
        super().__init__()
        self.fold, self.push, self.decay = fold, push, decay
        self.push_below, self.e_fold = push_below, e_fold

    def reset(self, player_id, seed=None):
        super().reset(player_id, seed)
        self._m = 0.0

    def act(self, obs, legal_mask, public_state=None):
        self._m = self.decay * self._m + (1 - self.decay) * float(obs[SIG_OPP])
        if (legal_mask[ACTION_EXIT] and self._m >= self.fold
                and float(obs[E_HAT]) < self.e_fold):
            return ACTION_EXIT
        a = _sig(self.push if self._m < self.push_below else 0.0)
        return a if legal_mask[a] else self.first_legal(legal_mask)


class DoveThreshold(Agent):
    """Never escalates; concedes once its own endurance drops below theta.

    theta = 1.0 reproduces the instant-fold variant, theta = 0.0 the calm
    one. The published Dove row sits between the two.
    """
    name = "Dove"

    def __init__(self, theta=0.6):
        super().__init__()
        self.theta = theta

    def act(self, obs, legal_mask, public_state=None):
        if legal_mask[ACTION_EXIT] and obs[E_HAT] < self.theta:
            return ACTION_EXIT
        a = _sig(0.0)
        return a if legal_mask[a] else self.first_legal(legal_mask)


VARIANTS = {
    "fold": DoveFold,
    "calm": DoveCalm,
}


def build_factories(master, dove="fold", bt_fold=0.5, dove_theta=None,
                    bt_push=0.75, bt_push_below=0.125, bt_e_fold=2.0):
    if dove_theta is not None:
        dove_cls = lambda: DoveThreshold(theta=dove_theta)
    else:
        dove_cls = VARIANTS[dove]
    f = {
        "Random": RandomAgent,
        "TitForTat": TitForTatAgent,
        "Hawk": Hawk,
        "Dove": dove_cls,
        "BayesianThreshold": lambda: BayesianThreshold(
            fold=bt_fold, push=bt_push, push_below=bt_push_below,
            e_fold=bt_e_fold),
    }
    if master.exists():
        f["MasterAgent"] = lambda: CheckpointAgent(str(master))
    return f


def board(factories, episodes, seed0, self_play):
    """Score every agent against the whole pool, both roles."""
    env = StrategicEnduranceEnv()
    seeds = [seed0 + k for k in range(episodes)]
    names = sorted(factories)
    as_I = {n: [] for n in names}
    as_U = {n: [] for n in names}
    for a, b in itertools.product(names, names):
        # Mirror matches: the reference agents are themselves members of the
        # reference pool, so each of them meets a copy of itself. A student
        # submission is not in that pool, so it never plays itself -- which
        # is exactly why Hawk lands below TitForTat on the published board
        # while an unopposed always-escalate strategy would top it.
        if a == b and (not self_play or a == "SUBMISSION"):
            continue
        agents = {IRAN: factories[a](), US: factories[b]()}
        for s in seeds:
            r = play_episode(env, agents, s)
            as_I[a].append(r["u"][IRAN])
            as_U[b].append(r["u"][US])
    rows = []
    for n in names:
        i, u = float(np.mean(as_I[n])), float(np.mean(as_U[n]))
        rows.append({"name": n, "score": 0.5 * (i + u), "iran": i, "us": u})
    rows.sort(key=lambda r: -r["score"])
    return rows


def render(rows, title):
    out = [title, ""]
    out.append("%-20s %8s %8s %8s   %s"
               % ("agent", "score", "Iran", "U.S.", "published (diff)"))
    for r in rows:
        line = "%-20s %8.1f %8.1f %8.1f" % (r["name"], r["score"], r["iran"],
                                            r["us"])
        if r["name"] in PUBLISHED:
            p = PUBLISHED[r["name"]]
            line += "   %5.1f (%+5.1f) %5.1f (%+5.1f) %5.1f (%+5.1f)" % (
                p[0], r["score"] - p[0], p[1], r["iran"] - p[1],
                p[2], r["us"] - p[2])
        out.append(line)
    known = [r for r in rows if r["name"] in PUBLISHED]
    if known:
        err = float(np.mean([abs(r["score"] - PUBLISHED[r["name"]][0])
                             for r in known]))
        out.append("")
        out.append("mean |score error| over %d published rows: %.2f"
                   % (len(known), err))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default=None)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=20_260_901)
    ap.add_argument("--dove", choices=list(VARIANTS), default="fold")
    ap.add_argument("--dove-theta", type=float, default=None)
    ap.add_argument("--bt-fold", type=float, default=0.5)
    ap.add_argument("--bt-push", type=float, default=0.75)
    ap.add_argument("--bt-push-below", type=float, default=0.125)
    ap.add_argument("--bt-e-fold", type=float, default=2.0)
    ap.add_argument("--both", action="store_true",
                    help="print with and without mirror matches")
    ap.add_argument("--self-play", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "analysis" / "board.txt"))
    a = ap.parse_args()

    f = build_factories(ROOT / "benchmarks" / "master_agent.pt",
                        dove=a.dove, bt_fold=a.bt_fold,
                        dove_theta=a.dove_theta,
                        bt_push=a.bt_push,
                        bt_push_below=a.bt_push_below,
                        bt_e_fold=a.bt_e_fold)
    if a.me:
        f["SUBMISSION"] = lambda: CheckpointAgent(a.me)

    blocks = []
    modes = [True, False] if a.both else [a.self_play]
    for sp in modes:
        rows = board(f, a.episodes, a.seed0, sp)
        blocks.append(render(
            rows, "dove=%s  bt_fold=%.2f  mirror matches %s  (%d eps, seed0=%d)"
            % (a.dove if a.dove_theta is None else "theta=%.2f" % a.dove_theta,
               a.bt_fold, "INCLUDED" if sp else "excluded",
               a.episodes, a.seed0)))
    io.open(a.out, "w", encoding="utf-8").write("\n\n".join(blocks) + "\n")
    sys.stdout.write("wrote " + a.out + "\n")


if __name__ == "__main__":
    main()
