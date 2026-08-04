#!/usr/bin/env python3
"""Section 5.5: confront the analytical propositions with data produced by
the trained agent.

The manual asks for the analytical statement of Sec. 4.8 to be turned into
an *observable statistic* and tested. We test four, exactly the four the
manual suggests:

  P1  the continuation gate    -- exit times as a function of the drawn type
  P2  the optimal signal       -- signal strength as a function of remaining
                                  endurance, and the per-stage signalling bill
  P3  constructed exit         -- exit decisions vs an open mediation window
  P4  self-trapping            -- accumulated commitments vs propensity to exit

Usage:
    python analysis/propositions.py runs/best.pt --episodes 300
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see import IRAN, PLAYERS, US                          # noqa: E402
from see.agents import BASELINES                            # noqa: E402
from see.agents.checkpoint import CheckpointAgent           # noqa: E402
from see.config import (ACTION_EXIT, ACTION_HOLD, HIGH_SIGNAL,  # noqa: E402
                        SIGNAL_LEVELS, canonical_config)
from see.env import StrategicEnduranceEnv, other            # noqa: E402

CFG = canonical_config()


def collect(ckpt, episodes, seed0=4242):
    """Play `episodes` episodes per (role, reference opponent) and record
    one row per stage per player."""
    sub = CheckpointAgent(ckpt)
    env = StrategicEnduranceEnv()
    rows, eps = [], []
    for opp_name, opp_cls in BASELINES.items():
        for role in (IRAN, US):
            for e in range(episodes):
                seed = seed0 + e
                opp = opp_cls()
                agents = ({role: sub, other(role): opp})
                obs, _ = env.reset(seed=seed)
                for pid in PLAYERS:
                    agents[pid].reset(pid, seed=seed * 2 + (pid == US))
                truth0 = env._info()["_true_state"]
                done, t = False, 0
                while not done:
                    public = env.public_state()
                    o_me = obs[role].copy()
                    K_me = 4 * o_me[4]
                    acts = {p: agents[p].act(obs[p], env.get_legal_mask(p),
                                             public) for p in PLAYERS}
                    a_me = acts[role]
                    forced = not env.get_legal_mask(role)[:ACTION_EXIT].any()
                    could_exit = bool(env.get_legal_mask(role)[ACTION_EXIT])
                    obs, rew, term, _, info = env.step(acts)
                    done = term[IRAN]
                    if a_me != ACTION_HOLD:
                        rows.append({
                            "role": role, "opp": opp_name, "t": t,
                            "e_hat": float(o_me[3]),
                            "K": float(K_me),
                            "med": float(o_me[8]),
                            "sig_opp": float(o_me[7]),
                            "action": int(a_me),
                            "sigma": (SIGNAL_LEVELS[a_me]
                                      if a_me < ACTION_EXIT else 0.0),
                            "exit": int(a_me == ACTION_EXIT),
                            "forced": int(forced),
                            "could_exit": int(could_exit),
                            "phi": float(info["phi"][role]),
                            "g": float(info["g"][role]),
                        })
                    t += 1
                truth = info["_true_state"]
                te = next((h["t"] for h in env.public_history
                           if h["exit"][role]), None)
                eps.append({
                    "role": role, "opp": opp_name,
                    "rho": float(truth0["rho"][role]),
                    "m_bar": float(truth0["m_bar"][role]),
                    "e0": float(truth0["e0"][role]),
                    "u": float(truth["utility"][role]),
                    "len": env.t,
                    "exit_t": te,
                    "exited": te is not None,
                })
    return rows, eps


def q_star(rows):
    """Empirical distribution of the continuation threshold
       q* = (g + phi - dX) / (B - X)   (paper Sec. 4.1)."""
    out = []
    for r in rows:
        p = CFG.players[r["role"]]
        # B and X reconstructed from the recorded state
        B = p.b0 * (p.b_base + p.b_e * min(r["e_hat"], 1.2) + p.b_rho * 0.6)
        X = p.x0 - p.lam * r["K"] - p.h * r["sig_opp"] + p.w_med * r["med"]
        K_next = CFG.delta_K * r["K"] + r["sigma"]
        dX = -p.lam * (K_next - r["K"])
        denom = B - X
        if denom > 1e-6:
            out.append((r["g"] + r["phi"] - dX) / denom)
    return np.array(out)


def report(rows, eps, out_path):
    R = {}
    print("=" * 74)
    print("P1  THE CONTINUATION GATE")
    print("=" * 74)
    q = q_star(rows)
    print(f"  empirical continuation threshold q* = (g+phi-dX)/(B-X):")
    print(f"    median {np.median(q):.3f}   mean {q.mean():.3f}   "
          f"90th pct {np.percentile(q, 90):.3f}")
    print(f"  -> a player should continue whenever it believes its chance of")
    print(f"     outlasting exceeds ~{np.median(q):.0%}. Exit is rational only")
    print(f"     when the race is nearly certainly lost.")
    R["q_star"] = {"median": float(np.median(q)), "mean": float(q.mean()),
                   "p90": float(np.percentile(q, 90))}

    # A stage counts as a *voluntary* exit opportunity only if staying was
    # also legal. When endurance runs out the engine leaves EXIT as the one
    # legal action, and counting those as choices biases every statistic
    # below: forced exits happen late, with endurance spent and the player's
    # own commitment stock K at its highest, which is exactly the direction
    # P4 predicts for voluntary exits. Mixing them in reverses P4's sign.
    vol = [r for r in rows if r["could_exit"] and not r["forced"]]
    ex = [r for r in vol if r["exit"]]
    print(f"\n  voluntary exits: {len(ex)}/{len(vol)} "
          f"({100*len(ex)/max(len(vol),1):.1f}% of stages where exit was legal)")
    if ex:
        print(f"    mean e_hat at exit      {np.mean([r['e_hat'] for r in ex]):.3f}")
        print(f"    mean e_hat when staying "
              f"{np.mean([r['e_hat'] for r in vol if not r['exit']]):.3f}")
    R["exit_rate"] = len(ex) / max(len(vol), 1)

    # exit time vs drawn type
    print("\n  exit time vs drawn type (quartiles of baseline resolve rho):")
    byq = collections.defaultdict(list)
    rhos = np.array([e["rho"] for e in eps])
    qs = np.percentile(rhos, [25, 50, 75])
    for e in eps:
        b = int(np.searchsorted(qs, e["rho"]))
        byq[b].append(e)
    print(f"    {'rho quartile':>14s} {'n':>5s} {'P(exit)':>9s} "
          f"{'mean exit t':>12s} {'mean u':>9s}")
    tab = {}
    for b in sorted(byq):
        v = byq[b]
        ts = [e["exit_t"] for e in v if e["exit_t"] is not None]
        pe = np.mean([e["exited"] for e in v])
        print(f"    {['Q1','Q2','Q3','Q4'][b]:>14s} {len(v):5d} {pe:9.3f} "
              f"{(np.mean(ts) if ts else float('nan')):12.2f} "
              f"{np.mean([e['u'] for e in v]):9.1f}")
        tab[f"Q{b+1}"] = {"n": len(v), "p_exit": float(pe),
                          "u": float(np.mean([e["u"] for e in v]))}
    R["by_rho"] = tab

    print("\n" + "=" * 74)
    print("P2  THE OPTIMAL SIGNAL IS INTERIOR")
    print("=" * 74)
    print(f"  {'sigma':>7s} {'stages':>8s} {'share':>7s} {'mean phi':>10s} "
          f"{'mean e_hat':>11s}")
    bysig = collections.defaultdict(list)
    for r in rows:
        if r["action"] < ACTION_EXIT:
            bysig[r["sigma"]].append(r)
    tot = sum(len(v) for v in bysig.values())
    tab = {}
    for s in SIGNAL_LEVELS:
        v = bysig.get(s, [])
        if not v:
            print(f"  {s:7.2f} {0:8d} {0.0:7.1%}")
            continue
        print(f"  {s:7.2f} {len(v):8d} {len(v)/tot:7.1%} "
              f"{np.mean([r['phi'] for r in v]):10.3f} "
              f"{np.mean([r['e_hat'] for r in v]):11.3f}")
        tab[str(s)] = {"share": len(v) / tot,
                       "phi": float(np.mean([r["phi"] for r in v])),
                       "e_hat": float(np.mean([r["e_hat"] for r in v]))}
    R["by_sigma"] = tab

    print("\n  signal strength vs remaining endurance (terciles of e_hat):")
    e_all = np.array([r["e_hat"] for r in rows if r["action"] < ACTION_EXIT])
    cuts = np.percentile(e_all, [33, 67])
    bins = collections.defaultdict(list)
    for r in rows:
        if r["action"] < ACTION_EXIT:
            bins[int(np.searchsorted(cuts, r["e_hat"]))].append(r["sigma"])
    for b in sorted(bins):
        lab = ["low e_hat", "mid e_hat", "high e_hat"][b]
        v = bins[b]
        print(f"    {lab:>12s}  mean sigma {np.mean(v):.3f}   "
              f"P(sigma>=0.75) {np.mean([x >= HIGH_SIGNAL for x in v]):.3f}")
    R["sigma_by_endurance"] = {
        ["low", "mid", "high"][b]: float(np.mean(bins[b])) for b in sorted(bins)}

    print("\n" + "=" * 74)
    print("P3  CONSTRUCTED EXIT: the mediation window")
    print("=" * 74)
    for flag, lab in ((1.0, "window OPEN "), (0.0, "window CLOSED")):
        v = [r for r in vol if r["med"] == flag]
        if v:
            pe = np.mean([r["exit"] for r in v])
            print(f"  {lab}: P(voluntary exit | can exit) = {pe:.4f}   "
                  f"(n = {len(v)})")
            R[f"exit_med_{int(flag)}"] = float(pe)
    if "exit_med_1" in R and "exit_med_0" in R and R["exit_med_0"] > 0:
        print(f"  ratio = {R['exit_med_1']/R['exit_med_0']:.2f}x")

    print("\n" + "=" * 74)
    print("P4  SELF-TRAPPING: own accumulated commitments")
    print("=" * 74)
    ks = np.array([r["K"] for r in vol])
    cuts = np.percentile(ks, [33, 67])
    tab = {}
    for b, lab in enumerate(("low K ", "mid K ", "high K")):
        v = [r for r in vol if int(np.searchsorted(cuts, r["K"])) == b]
        if v:
            pe = np.mean([r["exit"] for r in v])
            print(f"  {lab}: mean K = {np.mean([r['K'] for r in v]):.2f}   "
                  f"P(voluntary exit) = {pe:.4f}   (n = {len(v)})")
            tab[lab.strip()] = {"K": float(np.mean([r["K"] for r in v])),
                                "p_exit": float(pe)}
    R["by_K"] = tab

    pathlib.Path(out_path).write_text(json.dumps(R, indent=2))
    print(f"\nwrote {out_path}")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "props.json"))
    ap.add_argument("--role", choices=("I", "U", "both"), default="both",
                    help="restrict the statistics to one seat. The submitted "
                         "theory prices conceding on the U.S. seat only, so "
                         "splitting the seats shows which numbers that term "
                         "touches and which it leaves alone.")
    a = ap.parse_args()
    rows, eps = collect(a.checkpoint, a.episodes)
    if a.role != "both":
        rows = [r for r in rows if r["role"] == a.role]
        eps = [e for e in eps if e["role"] == a.role]
    print(f"stages recorded: {len(rows)}   episodes: {len(eps)}"
          f"   (seat: {a.role})\n")
    report(rows, eps, a.out)
