"""Extra scripted archetypes, used ONLY as training opponents and as a
strategy-landscape probe.

`docs/DESIGN.md` Sec. 11 says the public reference pool was reduced to
Random and TitForTat, and that the three archetypes used during the
instructor's own calibration (an always-escalate Hawk, an always-fold Dove
and a Bayesian-threshold agent) are picked up automatically by the trainer
if they are reintroduced. Training against only Random and TitForTat leaves
the learner with almost no pressure to behave sensibly against a *patient*
opponent, which is exactly the failure we measured.

Nothing here is under `see/`: the file lives in our own workspace, and the
training driver injects these classes into the trainer's opponent pool at
run time (in memory). The tournament is unaffected -- it runs the
instructor's engine and its own reference pool.
"""
from __future__ import annotations

import numpy as np

from see.agents.base import Agent
from see.config import ACTION_EXIT, SIGNAL_LEVELS

# obs indices (see see/env.py)
T_FRAC, RHO, M_HAT, E_HAT = 0, 1, 2, 3
K_OWN, K_OPP, SIG_OWN, SIG_OPP, MED = 4, 5, 6, 7, 8
HIGH_FREQ_OPP, MEAN_SIG_OPP = 11, 12


def _sig(level: float) -> int:
    return int(np.argmin([abs(level - s) for s in SIGNAL_LEVELS]))


class DoveAgent(Agent):
    """Maximal restraint: never escalates, never exits voluntarily.

    The point of the archetype is that restraint is *cheap* here -- raw
    costs scale with the opponent's posture, the commitment stock K stays
    at 0 so the exit value is never eroded, and a calm joint posture keeps
    the mediation window opening. It is the natural benchmark for "how much
    does escalation actually buy you?".
    """
    name = "Dove"

    def act(self, obs, legal_mask, public_state=None):
        a = _sig(0.0)
        return a if legal_mask[a] else self.first_legal(legal_mask)


class HawkAgent(Agent):
    """Maximal escalation: always sigma = 1.0, exits only when forced."""
    name = "Hawk"

    def act(self, obs, legal_mask, public_state=None):
        a = _sig(1.0)
        return a if legal_mask[a] else self.first_legal(legal_mask)


class PlateauAgent(Agent):
    """Signals at the credibility plateau sigma = 0.5 and never crosses the
    HIGH_SIGNAL = 0.75 threshold, so it is never locked out of exiting and
    never poisons the mediation chain. Exits when the endurance race is
    clearly lost."""
    name = "Plateau"

    def act(self, obs, legal_mask, public_state=None):
        if legal_mask[ACTION_EXIT] and obs[E_HAT] < 0.18:
            return ACTION_EXIT
        a = _sig(0.5 if obs[SIG_OPP] >= 0.5 else 0.25)
        return a if legal_mask[a] else self.first_legal(legal_mask)


class PatientAgent(Agent):
    """Calm attrition: restraint plus a late, endurance-conditioned exit.

    This is the strategy the Section 4 analysis points at -- keep K at
    zero so the exit option never erodes, keep the joint posture calm so
    mediation can open (X gains +20 when it does), and only concede when
    own endurance is genuinely spent."""
    name = "Patient"

    def act(self, obs, legal_mask, public_state=None):
        if legal_mask[ACTION_EXIT] and obs[E_HAT] < 0.12 and obs[MED] > 0.5:
            return ACTION_EXIT
        if legal_mask[ACTION_EXIT] and obs[E_HAT] < 0.06:
            return ACTION_EXIT
        a = _sig(0.0)
        return a if legal_mask[a] else self.first_legal(legal_mask)


class BayesThresholdAgent(Agent):
    """Escalates only while it believes it is winning the attrition race,
    using the closed-form collapse time of my_team/theory.py with the prior
    mean type for the opponent."""
    name = "BayesThreshold"

    def reset(self, player_id, seed=None):
        super().reset(player_id, seed)
        from my_team.theory import W, OTHER, _collapse_time
        self._W, self._OTHER, self._T = W, OTHER, _collapse_time

    def act(self, obs, legal_mask, public_state=None):
        W, OTHER, T = self._W, self._OTHER, self._T
        pid = self.player_id
        pi, pj = W[pid], W[OTHER[pid]]
        rho, mhat, ehat = obs[RHO], obs[M_HAT], obs[E_HAT]
        K_i, K_j, s_i, s_j = 4 * obs[K_OWN], 4 * obs[K_OPP], \
            obs[SIG_OWN], obs[SIG_OPP]
        e0 = pi["E0"] * (pi["e0_base"] + pi["e0_rho"] * rho
                         + pi["e0_m"] * mhat)
        mb = pi["m_lo"] + (pi["m_hi"] - pi["m_lo"]) * mhat
        raw_i = pi["c0"] + pi["kappa"] * s_j + pi["alpha"] * K_i \
            + pi["eta"] * s_i * s_j
        Ti = T(ehat, e0, mb, raw_i)
        # opponent evaluated at its prior mean type
        rj = pj["rho_a"] / (pj["rho_a"] + pj["rho_b"])
        mj = pj["m_a"] / (pj["m_a"] + pj["m_b"])
        e0j = pj["E0"] * (pj["e0_base"] + pj["e0_rho"] * rj + pj["e0_m"] * mj)
        mbj = pj["m_lo"] + (pj["m_hi"] - pj["m_lo"]) * mj
        raw_j = pj["c0"] + pj["kappa"] * s_i + pj["alpha"] * K_j \
            + pj["eta"] * s_i * s_j
        Tj = T(max(ehat, 0.05), e0j, mbj, raw_j)   # crude: assume similar wear
        if legal_mask[ACTION_EXIT] and Ti < 0.6 * Tj and obs[E_HAT] < 0.25:
            return ACTION_EXIT
        a = _sig(0.5 if Ti > Tj else 0.0)
        return a if legal_mask[a] else self.first_legal(legal_mask)


EXTRA_BASELINES = {
    "Dove": DoveAgent,
    "Hawk": HawkAgent,
    "Plateau": PlateauAgent,
    "Patient": PatientAgent,
    "BayesThreshold": BayesThresholdAgent,
}
