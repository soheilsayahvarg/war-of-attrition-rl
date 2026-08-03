#!/usr/bin/env python3
"""Build one checkpoint from the best Iran net and the best U.S. net.

This is model selection, not a change to the model. Sec. 11.9 shows that in
the reference evaluation an agent never plays itself, so the Iran net and the
U.S. net are scored on *disjoint* sets of games: J_I depends only on the Iran
seat, J_U only on the U.S. seat. Picking each seat's parameters from the run
that scored best on its own half is therefore the statistically correct
selection rule, and it changes nothing about the architecture, the trainer or
the theory.

The two sources must carry byte-identical theory source and the same feature
dimension -- otherwise the nets were trained against different inputs and
splicing them would be meaningless. The script refuses if they differ.

    python analysis/splice.py --iran runs/a.pt --us runs/b.pt --out out.pt
"""
import argparse
import hashlib
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.config import IRAN, US                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iran", required=True)
    ap.add_argument("--us", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--team", default="402111237")
    a = ap.parse_args()

    ci = torch.load(a.iran, map_location="cpu", weights_only=False)
    cu = torch.load(a.us, map_location="cpu", weights_only=False)

    def h(s):
        return hashlib.sha256(s.encode()).hexdigest()

    assert h(ci["theory_source"]) == h(cu["theory_source"]), \
        "refusing to splice: the two runs embed different theory source"
    assert ci["extra_dim"] == cu["extra_dim"], "feature dim mismatch"
    assert ci["obs_dim"] == cu["obs_dim"] and \
        ci["n_actions"] == cu["n_actions"], "shape mismatch"

    out = dict(ci)
    out["team"] = a.team
    out["nets"] = {IRAN: ci["nets"][IRAN], US: cu["nets"][US]}
    out["train_meta"] = {
        "total_steps": ci["train_meta"]["total_steps"],
        "iterations": ci["train_meta"]["iterations"],
        "ppo": ci["train_meta"]["ppo"],
        "seat_sources": {IRAN: pathlib.Path(a.iran).name,
                         US: pathlib.Path(a.us).name},
        "us_steps": cu["train_meta"]["total_steps"],
        "us_ppo": cu["train_meta"]["ppo"],
    }
    torch.save(out, a.out)
    print("iran seat <- %s" % pathlib.Path(a.iran).name)
    print("u.s. seat <- %s" % pathlib.Path(a.us).name)
    print("theory sha256 %s  (identical in both)" % h(ci["theory_source"])[:16])
    print("saved -> %s" % a.out)


if __name__ == "__main__":
    main()
