#!/usr/bin/env python3
"""Sharpen a checkpoint's action distribution by rescaling its own head.

WHY
---
`see/agents/checkpoint.py` constructs the tournament agent with
`sample=True`, hard-coded. So a policy is never evaluated at its argmax:
whatever probability mass PPO left on the second-best action is spent, every
stage, in the graded run. Entropy annealing (`--entropy-final`) attacks this
during training, but it trades against exploration and it is expensive --
each attempt is a fresh 700k-step run.

The same effect is available for free afterwards. `SEEPolicy.step` computes

    softmax( pi(h).masked_fill(~legal, -1e9) )

so multiplying the *actor head's* weight and bias by tau > 1 multiplies every
logit by tau, which is exactly a softmax temperature of 1/tau. Two properties
make this safe:

  * the ordering of the legal actions is unchanged, so the argmax policy --
    the thing the theory and the training actually produced -- is identical.
    Only the amount of noise around it moves.
  * masking happens after the head, and -1e9 * tau is still -inf in effect,
    so illegal actions stay illegal.

At tau = 1 this is the shipped checkpoint; as tau grows it approaches greedy
play. Nothing under see/ is touched: the architecture, the dimensions and the
state-dict keys are unchanged, so the output is an ordinary submission that
the stock loader reads. It is our own network's weights, rescaled.

Whether sharpening *helps* is an empirical question and not an obvious one --
in a game of incomplete information a deterministic agent is also a
predictable one, and the mixing may be load-bearing. Hence this script writes
the variants and analysis/official.py ranks them.

    python analysis/sharpen.py runs/of_s3.pt --taus 1.5 2 3 5
"""
import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sharpen(src: pathlib.Path, dst: pathlib.Path, tau: float) -> None:
    # weights_only=False: we need the embedded theory source string, which
    # the safe loader also returns, but train_meta may hold plain dicts.
    ck = torch.load(str(src), map_location="cpu", weights_only=False)
    nets = {}
    for pid, sd in ck["nets"].items():
        sd = {k: v.clone() for k, v in sd.items()}
        sd["pi.weight"] = sd["pi.weight"] * tau
        sd["pi.bias"] = sd["pi.bias"] * tau
        nets[pid] = sd
    ck["nets"] = nets
    meta = dict(ck.get("train_meta") or {})
    meta["logit_scale"] = tau
    ck["train_meta"] = meta
    torch.save(ck, str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[1.5, 2.0, 3.0, 5.0])
    a = ap.parse_args()

    src = ROOT / a.ckpt
    stem = src.stem
    for tau in a.taus:
        tag = ("%g" % tau).replace(".", "p")
        dst = src.with_name("%s_t%s.pt" % (stem, tag))
        sharpen(src, dst, tau)
        print("wrote %s (tau=%g)" % (dst.relative_to(ROOT), tau))


if __name__ == "__main__":
    main()
