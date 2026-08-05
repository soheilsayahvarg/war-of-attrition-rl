#!/usr/bin/env python3
"""Find a local stand-in for the two references we cannot reconstruct.

State of play: four of the six board references reproduce to the decimal
(Random, TitForTat, Hawk, MasterAgent). Dove and BayesianThreshold do not,
and no setting of their free parameters reproduces both uploads at once --
the real pair discriminates between our two checkpoints by 40 and 71
points, every reconstruction of them by under 10.

    board slot         upload 1        upload 2      spread
    Dove             144.4 / 135.9   104.9 /  64.6   -39.5 / -71.3
    BayesThreshold    77.6 /   9.4    60.9 /  15.0   -16.7 /  +5.6

So selection cannot lean on those two slots. What it *can* lean on is a
proxy: some locally available opponent that reproduces the same ordering
for the same reason. `analysis/sigprofile.py` says the two checkpoints
differ in exactly one way -- upload 1 signals ~0 and takes the exit when
it is losing, upload 2 pins sigma = 0.75 from stage 0 and never exits --
so the discriminating opponent should be one that is *passive but
patient*: it will not escalate, and it will not leave either.

`my_team/opponents.py` already has two of those (Patient, Plateau), and
neither was in the `board` pool the shipped checkpoint trained against.
This measures whether they order the two checkpoints the way the board's
soft slots did.

    python analysis/proxy_probe.py runs/board_reference.pt submission.pt
"""
import argparse
import io
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from see.agents.checkpoint import CheckpointAgent              # noqa: E402
from see.config import IRAN, US                                # noqa: E402
from see.env import StrategicEnduranceEnv                      # noqa: E402
from see.tournament.runner import play_episode                 # noqa: E402
from my_team.opponents import PatientAgent, PlateauAgent       # noqa: E402
from my_team.reference_pool import DoveAgent                   # noqa: E402

# how the two board slots we cannot rebuild ordered the two uploads
BOARD_SOFT = {
    "Dove": {"board_reference": (144.4, 135.9),
             "submission": (104.9, 64.6)},
    "BayesianThreshold": {"board_reference": (77.6, 9.4),
                          "submission": (60.9, 15.0)},
}

CANDIDATES = {
    "Patient": PatientAgent,
    "Plateau": PlateauAgent,
    "Dove(theta=0.75)": lambda: DoveAgent(theta=0.75),
    "Dove(theta=0.55)": lambda: DoveAgent(theta=0.55),
}


def score(me, factory, seeds, env):
    as_I = [play_episode(env, {IRAN: me, US: factory()}, s)["u"][IRAN]
            for s in seeds]
    as_U = [play_episode(env, {IRAN: factory(), US: me}, s)["u"][US]
            for s in seeds]
    return float(np.mean(as_I)), float(np.mean(as_U))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs=2, help="upload 1 then upload 2")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=20_260_901)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "proxy.txt"))
    a = ap.parse_args()

    env = StrategicEnduranceEnv()
    seeds = [a.seed0 + k for k in range(a.episodes)]
    me = [CheckpointAgent(str(ROOT / c)) for c in a.ckpts]
    names = [pathlib.Path(c).stem for c in a.ckpts]

    out = ["does any local opponent reproduce the board's soft-slot ordering?",
           "", "%-20s %19s %19s %17s" % ("opponent", names[0], names[1],
                                         "spread (2 - 1)")]
    for slot, obs in BOARD_SOFT.items():
        k = list(obs)
        d = (obs[k[1]][0] - obs[k[0]][0], obs[k[1]][1] - obs[k[0]][1])
        out.append("%-20s %9.1f %9.1f %9.1f %9.1f %8.1f %8.1f  <- BOARD"
                   % (slot, obs[k[0]][0], obs[k[0]][1],
                      obs[k[1]][0], obs[k[1]][1], d[0], d[1]))
    out.append("")
    for nm, fac in CANDIDATES.items():
        r = [score(m, fac, seeds, env) for m in me]
        out.append("%-20s %9.1f %9.1f %9.1f %9.1f %8.1f %8.1f"
                   % (nm, r[0][0], r[0][1], r[1][0], r[1][1],
                      r[1][0] - r[0][0], r[1][1] - r[0][1]))

    text = "\n".join(out)
    io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
