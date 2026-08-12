#!/usr/bin/env python3
"""Build a submission whose two role policies come from different runs.

WHY THIS IS AVAILABLE
---------------------
A submission is one checkpoint, but a checkpoint holds *two* networks --
`nets[IRAN]` and `nets[US]` -- because the engine trains one policy per
role. The tournament scores an agent as the mean over both seats and it
loads each seat's network independently: `CheckpointAgent` builds one
`SEEPolicy` per player id and `act` dispatches on `self.player_id`. The two
never interact.

Nothing in the assignment ties them together. The rules are: don't modify
`see/`, keep `theory.py` deterministic and self-contained, submit exactly
one checkpoint. A grafted checkpoint is still one checkpoint, of exactly the
standard architecture, carrying exactly one theory.py -- which is why this
script refuses to graft across differing theory sources. What changes is
only *which* of our own training runs each seat's weights came from.

WHY IT SHOULD HELP
------------------
The seat scores are wildly uneven and they are uneven in opposite
directions (analysis/head2head.py, seed 98765431):

    candidate     as Iran     as U.S.
    og_s2           126.8        37.5
    of_s3           113.1        51.3
    TitForTat       126.5        35.2

TitForTat's whole strength is the Iran seat and its whole weakness is the
U.S. seat; our official-pool runs are the mirror image. Each run is a
compromise between the two seats that no single training objective had to
make -- the seats are separate networks, so the compromise was never
required in the first place. Taking og_s2's Iran net and of_s3's U.S. net
should give roughly (126.8 + 51.3) / 2 against TitForTat's 80.9.

"Should" is doing real work in that sentence and the estimate is not a
measurement: the two seats meet the same opponents but a grafted pair has
never played together, and an opponent's behaviour in the Iran seat depends
on what our U.S. net does to it. analysis/head2head.py scores the result.

    python analysis/graft.py --iran runs/og_s2.pt --us runs/of_s3.pt \
        --out runs/graft_a.pt
"""
import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.config import IRAN, US                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iran", required=True, help="checkpoint for the Iran seat")
    ap.add_argument("--us", required=True, help="checkpoint for the U.S. seat")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck_i = torch.load(str(ROOT / a.iran), map_location="cpu",
                      weights_only=False)
    ck_u = torch.load(str(ROOT / a.us), map_location="cpu", weights_only=False)

    # The embedded theory source is what the loader executes to rebuild the
    # extra features, and there is only room for one copy. Grafting seats
    # whose features were computed by different theories would ship a
    # checkpoint whose Iran net was trained on inputs it will never see
    # again -- silently, since the dims would still match.
    if ck_i.get("theory_source") != ck_u.get("theory_source"):
        sys.exit("refusing to graft: the two checkpoints embed different "
                 "theory.py sources")
    if ck_i.get("extra_dim") != ck_u.get("extra_dim"):
        sys.exit("refusing to graft: extra feature dims differ")

    out = dict(ck_i)
    out["nets"] = {IRAN: ck_i["nets"][IRAN], US: ck_u["nets"][US]}
    meta = dict(out.get("train_meta") or {})
    meta["graft"] = {"iran": a.iran, "us": a.us}
    out["train_meta"] = meta
    torch.save(out, str(ROOT / a.out))
    print("wrote %s  (Iran <- %s, U.S. <- %s)" % (a.out, a.iran, a.us))


if __name__ == "__main__":
    main()
