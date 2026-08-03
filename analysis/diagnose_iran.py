#!/usr/bin/env python3
"""Why does TitForTat's Iran seat out-earn ours?

analysis/classpool.py shows the whole deficit sits in the Iran seat and is
flat in pool size.  This script takes the Iran seat apart opponent by
opponent and reports, next to each payoff, the three things that can explain
a gap: how often we leave voluntarily, how long the duel runs, and how much
endurance we still hold when it ends (which is what B is priced on).

It also reports the sampled-vs-greedy difference.  The tournament loads
checkpoints with sample=True, so a diffuse policy pays a tax at evaluation
time that a sharper policy would not.

    python analysis/diagnose_iran.py [--episodes 60]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent                  # noqa: E402
from see.agents.scripted import RandomAgent, TitForTatAgent        # noqa: E402
from see.config import ACTION_EXIT, IRAN, PLAYERS, US              # noqa: E402
from see.env import StrategicEnduranceEnv                          # noqa: E402


def run(env, iran, us, seeds):
    """Play the duel, tracking who left and in what shape."""
    out = {"u_I": [], "u_U": [], "len": [], "exit_I": [], "exit_U": [],
           "timeout": [], "sig_I": [], "ehat_I_end": []}
    for s in seeds:
        obs, _ = env.reset(seed=s)
        for role, ag in ((IRAN, iran), (US, us)):
            ag.reset(role, seed=s * 7919 + (1 if role == IRAN else 2))
        done, sigs, voluntary_I = False, [], False
        while not done:
            public = env.public_state()
            acts = {}
            for role, ag in ((IRAN, iran), (US, us)):
                mask = env.get_legal_mask(role)
                a = ag.act(obs[role], mask, public)
                if not mask[int(a)]:
                    a = int(np.argmax(mask))
                acts[role] = int(a)
            if acts[IRAN] == ACTION_EXIT:
                voluntary_I = True
            sigs.append(acts[IRAN])
            obs, rew, term, trunc, info = env.step(acts)
            done = term[IRAN] or trunc[IRAN]
        truth = info["_true_state"]
        out["u_I"].append(truth["utility"][IRAN])
        out["u_U"].append(truth["utility"][US])
        out["len"].append(env.t)
        out["exit_I"].append(float(voluntary_I))
        out["exit_U"].append(float(env.exited[US]))
        out["timeout"].append(float(not any(env.exited[p] for p in PLAYERS)))
        out["sig_I"].append(float(np.mean([min(a, 4) / 4.0 for a in sigs
                                           if a <= 4])) if sigs else 0.0)
        e0 = truth["e0"][IRAN] if "e0" in truth else None
        out["ehat_I_end"].append(
            truth["e"][IRAN] / e0 if e0 else float("nan"))
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--me", default=str(ROOT / "submission.pt"))
    a = ap.parse_args()

    seeds = [a.seed0 + k for k in range(a.episodes)]
    env = StrategicEnduranceEnv()

    opponents = [("Random", RandomAgent()), ("TitForTat", TitForTatAgent())]
    for nm, p in [("master", ROOT / "benchmarks" / "master_agent.pt"),
                  ("refspec_s3", ROOT / "runs" / "refspec_s3.pt"),
                  ("rich_s0", ROOT / "runs" / "rich_s0.pt")]:
        if p.exists():
            opponents.append((nm, CheckpointAgent(str(p))))

    mine_s = CheckpointAgent(a.me, sample=True)
    mine_g = CheckpointAgent(a.me, sample=False)
    tft = TitForTatAgent()

    hdr = ("%-14s %9s %8s %8s %8s %8s %8s"
           % ("vs (U.S. seat)", "u_I", "gap", "len", "exit_I", "exitU", "sig_I"))
    print("\nIRAN SEAT — sampled checkpoint vs TitForTat, same opponents")
    print(hdr)
    rows = []
    for nm, opp in opponents:
        m = run(env, mine_s, opp, seeds)
        t = run(env, tft, opp, seeds)
        rows.append({"opp": nm, "me": m, "tft": t})
        print("%-14s %9.1f %8.1f %8.1f %8.2f %8.2f %8.2f"
              % (nm, m["u_I"], m["u_I"] - t["u_I"], m["len"],
                 m["exit_I"], m["exit_U"], m["sig_I"]))
        print("%-14s %9.1f %8s %8.1f %8.2f %8.2f %8.2f"
              % ("   (TitForTat)", t["u_I"], "", t["len"],
                 t["exit_I"], t["exit_U"], t["sig_I"]))

    print("\nsampled vs greedy (same checkpoint, Iran seat)")
    print("%-14s %9s %9s %9s" % ("vs", "sampled", "greedy", "delta"))
    for nm, opp in opponents:
        s = run(env, mine_s, opp, seeds)["u_I"]
        g = run(env, mine_g, opp, seeds)["u_I"]
        print("%-14s %9.1f %9.1f %+9.1f" % (nm, s, g, g - s))

    (ROOT / "analysis" / "diagnose_iran.json").write_text(
        json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
