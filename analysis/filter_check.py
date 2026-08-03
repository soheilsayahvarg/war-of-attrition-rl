#!/usr/bin/env python3
"""Diagnostic: does the theory's Bayesian filter actually track the
opponent's hidden endurance and type?

The filter in my_team/theory.py claims that, because the cost law is
public, a posterior over the opponent's *endurance trajectory* reduces to a
posterior over the two-dimensional structural type. This script checks the
claim against ground truth (env.info["_true_state"], which agents never
see) and reports calibration by stage.

    python analysis/filter_check.py [--episodes 300]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see import IRAN, PLAYERS, US, StrategicEnduranceEnv   # noqa: E402
from see.agents import BASELINES                            # noqa: E402
from see.env import other                                   # noqa: E402
from see.training.theory_api import load_theory             # noqa: E402


def run(n_episodes: int, seed0: int = 1234):
    theory, _ = load_theory(str(ROOT / "my_team" / "theory.py"))
    env = StrategicEnduranceEnv()
    rng = np.random.default_rng(seed0)
    names = list(BASELINES)

    rows = []
    for ep in range(n_episodes):
        aI = BASELINES[names[rng.integers(len(names))]]()
        aU = BASELINES[names[rng.integers(len(names))]]()
        seed = seed0 + ep
        obs, info = env.reset(seed=seed)
        aI.reset(IRAN, seed=seed * 2 + 1)
        aU.reset(US, seed=seed * 2 + 2)
        agents = {IRAN: aI, US: aU}
        for p in PLAYERS:
            theory.on_episode_start(p)

        done, t = False, 0
        while not done:
            public = env.public_state()
            truth = env._info()["_true_state"]
            for pid in PLAYERS:
                j = other(pid)
                f = theory.extra_features(pid, obs[pid], public)
                rows.append({
                    "ep": ep, "t": t, "pid": pid,
                    "post_ehat_j": float(f[0]),
                    "post_sd": float(f[1]),
                    "post_rho_j": float(f[3]),
                    "post_mhat_j": float(f[4]),
                    "q_hat": float(f[8]),
                    "gate": float(f[9]),
                    "true_ehat_j": float(truth["e"][j] / truth["e0"][j]),
                    "true_rho_j": float(truth["rho"][j]),
                })
            acts = {p: agents[p].act(obs[p], env.get_legal_mask(p), public)
                    for p in PLAYERS}
            obs, _r, term, _tr, info = env.step(acts)
            done = term[IRAN]
            t += 1

    return rows


def report(rows):
    import collections
    e_err, rho_err = [], []
    by_stage = collections.defaultdict(list)
    q_bins = collections.defaultdict(list)

    for r in rows:
        de = r["post_ehat_j"] - r["true_ehat_j"]
        e_err.append(de)
        rho_err.append(r["post_rho_j"] - r["true_rho_j"])
        by_stage[min(r["t"], 12)].append(abs(de))

    e_err = np.array(e_err)
    rho_err = np.array(rho_err)
    print(f"observations: {len(rows)}")
    print()
    print("posterior mean of opponent e_hat  vs  ground truth")
    print(f"  bias  {e_err.mean():+.4f}   MAE {np.abs(e_err).mean():.4f} "
          f"  RMSE {np.sqrt((e_err**2).mean()):.4f}")
    print("posterior mean of opponent rho_bar  vs  ground truth")
    print(f"  bias  {rho_err.mean():+.4f}   MAE {np.abs(rho_err).mean():.4f}"
          f"   RMSE {np.sqrt((rho_err**2).mean()):.4f}")
    print()
    print("  stage    n     MAE(e_hat_j)")
    for t in sorted(by_stage):
        v = np.array(by_stage[t])
        lab = f"{t}" if t < 12 else "12+"
        print(f"  {lab:>5s}  {len(v):5d}   {v.mean():.4f}")

    # naive baseline: always predict the prior mean
    prior = np.mean([r["true_ehat_j"] for r in rows])
    naive = np.abs(np.array([r["true_ehat_j"] for r in rows]) - prior).mean()
    print()
    print(f"naive constant-prediction MAE: {naive:.4f}  "
          f"-> filter reduces error by "
          f"{100*(1-np.abs(e_err).mean()/naive):.1f}%")
    return {
        "n": len(rows),
        "ehat_bias": float(e_err.mean()),
        "ehat_mae": float(np.abs(e_err).mean()),
        "ehat_rmse": float(np.sqrt((e_err ** 2).mean())),
        "rho_mae": float(np.abs(rho_err).mean()),
        "naive_mae": float(naive),
        "mae_by_stage": {str(t): float(np.mean(by_stage[t]))
                         for t in sorted(by_stage)},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "filter_check.json"))
    a = ap.parse_args()
    rows = run(a.episodes)
    summary = report(rows)
    pathlib.Path(a.out).write_text(json.dumps(summary, indent=2))
    np.save(ROOT / "analysis" / "filter_rows.npy",
            np.array([(r["t"], r["post_ehat_j"], r["true_ehat_j"],
                       r["post_rho_j"], r["true_rho_j"], r["q_hat"])
                      for r in rows], dtype=np.float32))
    print(f"\nwrote {a.out}")
