"""Endurance-race theory: an explicit Bayesian filter over the opponent's
type, plus a potential-based training signal built from the paper's two
terminal values.

WHY THIS WORKS (the one idea behind the whole file)
---------------------------------------------------
In this game the endurance state e_j is *latent*, but the law that moves it
is *public*:

    g_j^t = ramp(t) * (c0_j + kappa_j*sigma_i + alpha_j*K_j + eta*sigma_i*sigma_j)
            / [ m_bar_j * (0.35 + 0.65 * e_j/e0_j) ]

Everything on the right except (m_bar_j, e0_j) is either a published
constant or part of the public history. So conditional on a hypothesised
structural type theta_j = (rho_bar_j, m_bar_j), the whole endurance
trajectory of the opponent is pinned down (up to the small shock xi).

That collapses the filtering problem: a posterior over the *trajectory* is
just a posterior over the two-dimensional type. We carry a weighted grid
over theta_j and update it from three sources, each of which is one of the
manual's Section 4 questions made computable:

  (S1) SURVIVAL  (Sec. 4.3, "the information contained in the mere fact
       that the opponent has not exited yet"). e_j <= 0 forces an exit.
       The opponent is still here, so every grid point whose simulated
       endurance would already have hit zero is *impossible*. This
       likelihood is exact and needs no model of the opponent's strategy.

  (S2) COSTLY SIGNALS  (Sec. 4.4, single-crossing). phi_j is cheaper for
       high rho_bar / high e_hat / high m_hat, and d^2 phi/(d sigma d e) < 0,
       so a sustained loud posture is evidence of a strong type. We use the
       quantal-response likelihood exp(-BETA * phi_j(sigma_j)) on the
       stages where j actually *chose* a signal (HOLD is forced, hence
       uninformative, and is skipped).

  (S3) THE PRIOR, which is published in see/config.py and asymmetric
       between the two roles.

  (S4) COMMITMENT TYPES. (S1)-(S3) all learn about theta_j. They are blind
       to an opponent whose action is a fixed *reply rule*, because such a
       rule is independent of theta_j and therefore never separates. What
       identifies it is the rule itself: sigma_j(t) = sigma_i(t-1), stage
       after stage. We count exact replications; three in a row is the
       trigger. This matters because a commitment type is the one opponent
       against whom maximising our own utility and maximising our standing
       against it pull in opposite directions -- see `shaping`.

From that posterior we build the decision statistics of Sections 4.1-4.6:
a closed-form collapse time, the probability of winning the attrition race,
the continuation gate, the erosion rate of the exit option, and the
mediation option value.

Closed-form collapse time (used for the race statistics)
-------------------------------------------------------
With postures held fixed, R := raw flow cost is constant and, writing
x = e/e0, the endurance law is

    dx/dt = -R / ( e0 * m_bar * (0.35 + 0.65 x) )

    =>   integral_0^x (0.35 + 0.65 u) du = (R / (e0*m_bar)) * T

    =>   T_collapse = e0 * m_bar * (0.35 x + 0.325 x^2) / R.

This is the quantitative face of the paper's "compounding fragility":
capacity m_i^t itself decays with e, so the same raw pressure bites harder
the more worn down you already are, and T is *convex* in remaining
endurance.

Everything here is deterministic given (obs, public_state); nothing reads
the opponent's private state.
"""
import numpy as np

from see.training.theory_api import TheorySpec

# --------------------------------------------------------------------- #
# published world constants (mirror of see/config.py; imported when the  #
# package is available so the two can never silently drift apart)        #
# --------------------------------------------------------------------- #
_FALLBACK = {
    "I": dict(rho_a=5.0, rho_b=2.0, m_a=2.0, m_b=2.0, m_lo=0.8, m_hi=1.7,
              c0=1.4, kappa=3.0, alpha=0.8, eta=2.5, b0=175.0, x0=26.0),
    "U": dict(rho_a=3.0, rho_b=3.0, m_a=3.0, m_b=2.0, m_lo=0.9, m_hi=1.8,
              c0=1.8, kappa=2.5, alpha=1.3, eta=2.5, b0=175.0, x0=28.0),
}
_COMMON = dict(E0=100.0, e0_base=0.50, e0_rho=0.35, e0_m=0.15, m_floor=0.35,
               phi0=1.2, phi_w=0.40, phi_v=0.50, lam=5.0, h=8.0, w_med=20.0,
               b_base=0.50, b_e=0.30, b_rho=0.20)
_GLOBAL = dict(T_bar=40, delta_K=0.75, ramp_T=4, p_open_calm=0.35,
               p_open_hot=0.05, calm_thresh=0.50)


def _world():
    """Read the canonical parameters, falling back to the literals above."""
    try:
        from see.config import canonical_config
        cfg = canonical_config()
        out = {}
        for pid in ("I", "U"):
            pp = cfg.players[pid]
            out[pid] = dict(
                rho_a=pp.rho_a, rho_b=pp.rho_b, m_a=pp.m_a, m_b=pp.m_b,
                m_lo=pp.m_lo, m_hi=pp.m_hi, c0=pp.c0, kappa=pp.kappa,
                alpha=pp.alpha, eta=pp.eta, b0=pp.b0, x0=pp.x0, E0=pp.E0,
                e0_base=pp.e0_base, e0_rho=pp.e0_rho, e0_m=pp.e0_m,
                m_floor=pp.m_floor, phi0=pp.phi0, phi_w=pp.phi_w,
                phi_v=pp.phi_v, lam=pp.lam, h=pp.h, w_med=pp.w_med,
                b_base=pp.b_base, b_e=pp.b_e, b_rho=pp.b_rho)
        out["_"] = dict(T_bar=cfg.T_bar, delta_K=cfg.delta_K,
                        ramp_T=cfg.ramp_T, p_open_calm=cfg.p_open_calm,
                        p_open_hot=cfg.p_open_hot,
                        calm_thresh=cfg.calm_thresh)
        return out
    except Exception:                                   # pragma: no cover
        out = {pid: dict(_FALLBACK[pid], **_COMMON) for pid in ("I", "U")}
        out["_"] = dict(_GLOBAL)
        return out


W = _world()
OTHER = {"I": "U", "U": "I"}

# grid resolution over theta_j = (rho_bar, m_bar); 11 x 9 = 99 points
N_RHO, N_M = 11, 9
# quantal-response temperature for the single-crossing signal likelihood
BETA_SIG = 0.45
# endurance below this fraction of e0 counts as "collapse imminent"
COLLAPSE_BAND = 0.15
# potential weights for the training-time shaping term (see `shaping`)
W_X, W_B, GAMMA = 0.50, 0.10, 0.995

# --- (S4) commitment types: a posterior, not a trigger ----------------- #
# A type whose action is a fixed, history-independent reply carries no
# information about theta_j through the single-crossing channel, so channels
# (S1)-(S3) never separate it. What identifies it is the *rule* itself:
# sigma_j(t) = sigma_i(t-1), stage after stage.
#
# We carry an explicit posterior over the binary hypothesis "j follows a
# mirror rule" rather than a counter with a threshold. A threshold is
# exactly the wrong object here: it hands the learner a deadline to beat,
# and our first attempt at this failed for that reason -- the agent simply
# quit one stage earlier than the trigger. A posterior has no deadline. It
# starts at the share of mirror types among the published reference
# opponents and moves on the very first observation.
COMMIT_PRIOR = 0.35        # Random and TitForTat: one of the two mirrors
COMMIT_LR = 4.0            # odds multiplier per exact replication
# price charged at training time for conceding, per unit of prize handed to
# the opponent (see `shaping`; units are raw utility). Once the term is gated
# on the posterior it costs nothing against an opponent that is not a mirror,
# so there is no longer a reason to restrict it to one seat: both seats hand
# the same prize away when they concede to an identified commitment type.
CONCEDE_PRICE = 0.0
CONCEDE_SEAT = None         # None = both seats

ACT_EXIT, ACT_HOLD = 5, 6


def _beta_pdf(x, a, b):
    """Unnormalised Beta density (normalisation cancels in the weights)."""
    return np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)


def _phi(sigma, rho, e_hat, m_hat, p):
    """Signaling cost phi_i(sigma; theta_i, e_i)  (paper Sec. 6)."""
    d = ((p["phi_w"] + (1 - p["phi_w"]) * rho)
         * (p["phi_w"] + (1 - p["phi_w"]) * np.minimum(e_hat, 1.0))
         * (p["phi_v"] + (1 - p["phi_v"]) * m_hat))
    return p["phi0"] * sigma * sigma / np.maximum(d, 1e-6)


def _collapse_time(e_hat, e0, m_bar, raw):
    """Closed-form stages-to-forced-exit at a frozen posture (see header)."""
    x = np.clip(e_hat, 0.0, 1.2)
    return e0 * m_bar * (0.35 * x + 0.325 * x * x) / np.maximum(raw, 1e-6)


class EnduranceRaceTheory(TheorySpec):
    """Beliefs from Sec. 4.3-4.4; decision statistics from Sec. 4.1-4.6."""

    name = "endurance-race-filter+behaviour"
    extra_feature_dim = 19

    # ----------------------------------------------------------------- #
    # grid construction                                                  #
    # ----------------------------------------------------------------- #
    def __init__(self):
        self._grid = {pid: self._build_grid(pid) for pid in ("I", "U")}
        self._st = {}

    @staticmethod
    def _build_grid(pid):
        """Prior grid over the *opponent's* structural type theta_j."""
        p = W[OTHER[pid]]
        r = (np.arange(N_RHO) + 0.5) / N_RHO           # rho_bar in (0,1)
        mh = (np.arange(N_M) + 0.5) / N_M              # m_hat  in (0,1)
        R, MH = np.meshgrid(r, mh, indexing="ij")
        R, MH = R.ravel(), MH.ravel()
        w = (_beta_pdf(R, p["rho_a"], p["rho_b"])
             * _beta_pdf(MH, p["m_a"], p["m_b"]))
        w = w / w.sum()
        m_bar = p["m_lo"] + (p["m_hi"] - p["m_lo"]) * MH
        e0 = p["E0"] * (p["e0_base"] + p["e0_rho"] * R + p["e0_m"] * MH)
        return {"rho": R, "m_hat": MH, "m_bar": m_bar, "e0": e0,
                "logw0": np.log(np.maximum(w, 1e-300))}

    # ----------------------------------------------------------------- #
    # per-episode filter state                                           #
    # ----------------------------------------------------------------- #
    def on_episode_start(self, player_id):
        self._reset(player_id if player_id in ("I", "U") else "I")

    def _reset(self, pid):
        g = self._grid[pid]
        self._st[pid] = {"cursor": 0,
                         "logw": g["logw0"].copy(),
                         "e": g["e0"].copy(),
                         "K_opp": 0.0,
                         # (S4) posterior that j follows a mirror rule, plus
                         # the statistics `shaping` needs at the stage the
                         # choice is actually made
                         "prev_sig_own": None,
                         "p_commit": COMMIT_PRIOR,
                         "prize_j": 0.9,
                         "q_hat": 0.5,
                         # (S5) raw sufficient statistics of the opponent's
                         # *signal stream*; see the note above features 16-18
                         "n_free": 0,
                         "sum_sig_j": 0.0,
                         "sum_sig_j2": 0.0,
                         "max_sig_j": 0.0,
                         "n_zero_j": 0}

    # ----------------------------------------------------------------- #
    # (1) BELIEF FEATURES                                                #
    # ----------------------------------------------------------------- #
    def extra_features(self, player_id, obs, public_state):
        pid = player_id if player_id in ("I", "U") else "I"
        opp = OTHER[pid]
        obs = np.asarray(obs, dtype=np.float64)
        hist = (public_state or {}).get("history", []) or []

        st = self._st.get(pid)
        if st is None or st["cursor"] > len(hist):
            self._reset(pid)
        self._advance(pid, hist)
        st = self._st[pid]

        g = self._grid[pid]
        w = self._weights(st)

        # ---- own state, reconstructed exactly from the observation ---- #
        pi, pj = W[pid], W[opp]
        rho_i, mhat_i, ehat_i = obs[1], obs[2], obs[3]
        K_i, K_j = 4.0 * obs[4], 4.0 * obs[5]
        sig_i, sig_j, med = obs[6], obs[7], obs[8]
        t = float(np.round(obs[0] * W["_"]["T_bar"]))
        e0_i = pi["E0"] * (pi["e0_base"] + pi["e0_rho"] * rho_i
                           + pi["e0_m"] * mhat_i)
        mbar_i = pi["m_lo"] + (pi["m_hi"] - pi["m_lo"]) * mhat_i
        ramp = min(1.0, (t + 1.0) / W["_"]["ramp_T"])

        # ---- posterior summaries over the opponent -------------------- #
        ehat_j = np.clip(st["e"] / g["e0"], 0.0, 1.2)
        m_ehat_j = float(w @ ehat_j)
        sd_ehat_j = float(np.sqrt(max(float(w @ (ehat_j - m_ehat_j) ** 2), 0.0)))
        p_collapse = float(w @ (ehat_j <= COLLAPSE_BAND))
        m_rho_j = float(w @ g["rho"])
        m_mhat_j = float(w @ g["m_hat"])

        # ---- the attrition race (Sec. 4.1) ---------------------------- #
        raw_i = (pi["c0"] + pi["kappa"] * sig_j + pi["alpha"] * K_i
                 + pi["eta"] * sig_i * sig_j)
        raw_j = (pj["c0"] + pj["kappa"] * sig_i + pj["alpha"] * K_j
                 + pj["eta"] * sig_i * sig_j)
        T_i = float(_collapse_time(ehat_i, e0_i, mbar_i, raw_i))
        T_j = _collapse_time(ehat_j, g["e0"], g["m_bar"], raw_j)
        q_hat = float(w @ (T_j < T_i))                  # P(win the race)
        T_race = float(w @ np.minimum(T_j, T_i))

        # ---- the two terminal values (paper Sec. 7) ------------------- #
        B_i = pi["b0"] * (pi["b_base"] + pi["b_e"] * min(ehat_i, 1.2)
                          + pi["b_rho"] * rho_i)
        X_i = pi["x0"] - pi["lam"] * K_i - pi["h"] * sig_j + pi["w_med"] * med

        # ---- per-stage burn, and the continuation gate (Sec. 4.1) ----- #
        m_it = mbar_i * (pi["m_floor"]
                         + (1 - pi["m_floor"]) * max(ehat_i, 0.0))
        g_i = ramp * raw_i / max(m_it, 1e-6)
        phi_i = float(_phi(sig_i, rho_i, ehat_i, mhat_i, pi))
        gate = (q_hat * (B_i - X_i) - T_race * (g_i + phi_i)) / 175.0

        # ---- credibility-flexibility price (Sec. 4.5-4.6) ------------- #
        # exit value next stage if the current posture is simply held
        K_i_next = W["_"]["delta_K"] * K_i + sig_i
        X_next = (pi["x0"] - pi["lam"] * K_i_next - pi["h"] * sig_j
                  + pi["w_med"] * med)
        erosion = (X_next - X_i) / 46.0
        # option value of restraint: chance the mediation window opens
        joint = sig_i + sig_j
        p_open = (W["_"]["p_open_calm"] if joint <= W["_"]["calm_thresh"]
                  else W["_"]["p_open_hot"])
        med_option = (1.0 - med) * p_open * pi["w_med"] / 46.0
        # probability that escalating to sigma >= 0.75 locks me out of
        # exiting next stage (it does unless the opponent also goes high)
        p_lock = 1.0 - float(np.clip(obs[11], 0.0, 1.0))

        # ---- commitment type, and the price of conceding --------------- #
        p_commit = float(st["p_commit"])
        # what quitting right now would hand the opponent: its outlasting
        # prize, priced on the endurance our posterior says it still holds.
        B_j = pj["b0"] * (pj["b_base"] + pj["b_e"] * np.minimum(ehat_j, 1.2)
                          + pj["b_rho"] * g["rho"])
        prize_j = float(w @ B_j) / 175.0
        st["prize_j"], st["q_hat"] = prize_j, q_hat

        # ---- how the opponent *behaves*, as opposed to what it is ------ #
        # Every feature above summarises the posterior over the opponent's
        # structural type theta_j = (rho_bar, m_bar) -- how much endurance it
        # has and how fast it burns. Three uploads showed that this is not
# enough. The U.S. seat scored 113.3 against Random by pinning
        # sigma = 0.75 and never exiting, and 64.6 against Dove by doing the
        # same thing; the checkpoint that fixed Dove (113.1) dropped Random
        # to 62.1. Two different opponent pools were tried and neither let
        # one policy do both (analysis/select2.py), which is evidence that
        # the limit is the representation, not the training distribution.
        #
        # The reason is visible once stated: Random and a patient opponent
        # differ almost entirely in the *variability* of their signal, and
        # nothing above measures variability. Random draws uniformly over
        # the five levels (sd ~ 0.35, reaches sigma = 1 early, rarely quiet);
        # Dove and Patient sit at sigma = 0 forever (sd = 0, max 0, always
        # quiet). The type filter cannot separate them because they are not
        # different *types* -- they are different *rules*.
        #
        # So these three are deliberately raw sample statistics of the
        # opponent's signal stream rather than posterior quantities. They add
        # no theory; they add the one axis the theory does not span.
        n_free = max(int(st["n_free"]), 0)
        if n_free > 0:
            mean_sig_j = st["sum_sig_j"] / n_free
            var_sig_j = max(st["sum_sig_j2"] / n_free - mean_sig_j ** 2, 0.0)
            sd_sig_j = float(np.sqrt(var_sig_j))
            frac_zero_j = st["n_zero_j"] / n_free
        else:
            # nothing observed yet: "no evidence" must not look like
            # "evidence of a quiet opponent", so sit at the neutral point
            sd_sig_j, frac_zero_j = 0.0, 0.5
        max_sig_j = float(st["max_sig_j"])

        f = np.array([
            m_ehat_j,                       # 0  E[e_hat_j]
            sd_ehat_j,                      # 1  posterior spread
            p_collapse,                     # 2  P(opponent near forced exit)
            m_rho_j,                        # 3  E[rho_bar_j]
            m_mhat_j,                       # 4  E[m_hat_j]
            ehat_i - m_ehat_j,              # 5  endurance edge
            min(float(w @ T_j), 40.0) / 20.0,               # 6  E[T_j]
            min(T_i, 40.0) / 20.0,          # 7  own stages to collapse
            q_hat,                          # 8  P(outlast opponent)
            gate,                           # 9  continuation gate
            X_i / 46.0,                     # 10 exit-option health
            erosion,                        # 11 flexibility price
            med_option,                     # 12 mediation option value
            p_lock,                         # 13 self-trap risk of sigma>=.75
            p_commit,                       # 14 P(opponent is a commitment type)
            prize_j,                        # 15 prize handed over by quitting
            sd_sig_j,                       # 16 how erratic the opponent is
            max_sig_j,                      # 17 has it ever escalated at all
            frac_zero_j,                    # 18 how often it stays silent
        ], dtype=np.float32)
        return np.nan_to_num(f, nan=0.0, posinf=3.0, neginf=-3.0)

    # ----------------------------------------------------------------- #
    # filter update                                                      #
    # ----------------------------------------------------------------- #
    def _advance(self, pid, hist):
        """Fold every unprocessed public stage into the posterior."""
        st, opp = self._st[pid], OTHER[pid]
        g, pj = self._grid[pid], W[opp]
        while st["cursor"] < len(hist):
            h = hist[st["cursor"]]
            t = int(h.get("t", st["cursor"]))
            sig = h.get("sigma", {})
            sig_i = float(sig.get(pid, 0.0))
            sig_j = float(sig.get(opp, 0.0))
            act_j = int(h.get("action", {}).get(opp, ACT_HOLD))
            K_j = st["K_opp"]

            e_hat = np.clip(st["e"] / g["e0"], 0.0, 1.2)

            # (S1) survival: e_j <= 0 forces EXIT, so a still-standing
            #      opponent rules those types out exactly.
            if act_j != ACT_EXIT:
                st["logw"] = np.where(st["e"] > 0.0, st["logw"], -np.inf)

            # (S2) single-crossing: a freely chosen loud posture is cheap
            #      only for strong types.
            if act_j < ACT_EXIT and sig_j > 0.0:
                st["logw"] = st["logw"] - BETA_SIG * _phi(
                    sig_j, g["rho"], e_hat, g["m_hat"], pj)

            # deterministic endurance transition under the public cost law
            ramp = min(1.0, (t + 1.0) / W["_"]["ramp_T"])
            m_t = g["m_bar"] * (pj["m_floor"] + (1 - pj["m_floor"])
                                * np.maximum(e_hat, 0.0))
            raw = (pj["c0"] + pj["kappa"] * sig_i + pj["alpha"] * K_j
                   + pj["eta"] * sig_j * sig_i)
            st["e"] = st["e"] - ramp * raw / np.maximum(m_t, 1e-6)

            # (S4) commitment type. A reply rule sigma_j(t) = sigma_i(t-1)
            #      is *behavioural* evidence, not type evidence: it survives
            #      every update above because it is independent of theta_j.
            #      We keep a proper posterior on the binary hypothesis. A
            #      mirror reproduces our previous signal with probability 1,
            #      anything else does so only by coincidence, so an exact
            #      replication multiplies the odds by COMMIT_LR and a single
            #      mismatch is decisive against. Only stages where j freely
            #      chose a signal count (HOLD is forced, hence uninformative).
            #      One caveat decides whether this works at all. The test
            #      "did j replay our last signal" is *degenerate* when that
            #      signal was zero: a quiet opponent that always plays 0
            #      trivially reproduces our 0 and looks exactly like a
            #      mirror. Measured consequence of ignoring this: the U.S.
            #      seat stopped conceding to everybody and lost 20 points of
            #      its own score in a class-sized field
            #      (analysis/classpool.py). So a stage only counts as
            #      evidence when there was something to copy, sigma_i > 0.
            informative = (act_j < ACT_EXIT
                           and st["prev_sig_own"] is not None
                           and st["prev_sig_own"] > 0.0)
            if informative:
                if abs(sig_j - st["prev_sig_own"]) < 1e-9:
                    o = st["p_commit"] / max(1.0 - st["p_commit"], 1e-9)
                    o *= COMMIT_LR
                    st["p_commit"] = o / (1.0 + o)
                else:
                    st["p_commit"] = 0.0
            st["prev_sig_own"] = sig_i

            # (S5) sufficient statistics of the opponent's signal stream.
            #      Only freely chosen signals count, for the same reason (S4)
            #      excludes HOLD: a forced move is not a choice and carries no
            #      information about how the opponent behaves.
            if act_j < ACT_EXIT:
                st["n_free"] += 1
                st["sum_sig_j"] += sig_j
                st["sum_sig_j2"] += sig_j * sig_j
                st["max_sig_j"] = max(st["max_sig_j"], sig_j)
                if sig_j <= 1e-9:
                    st["n_zero_j"] += 1

            st["K_opp"] = W["_"]["delta_K"] * K_j + sig_j
            st["cursor"] += 1

    @staticmethod
    def _weights(st):
        lw = st["logw"]
        finite = np.isfinite(lw)
        if not np.any(finite):                   # posterior wiped out:
            return np.full(lw.size, 1.0 / lw.size)   # fall back to uniform
        lw = lw - np.max(lw[finite])
        w = np.exp(np.clip(lw, -700.0, 0.0))
        w = np.where(finite, w, 0.0)
        s = w.sum()
        return w / s if s > 0 else np.full(lw.size, 1.0 / lw.size)

    # ----------------------------------------------------------------- #
    # (2) REWARD SHAPING - training only, potential-based                #
    # ----------------------------------------------------------------- #
    def shaping(self, player_id, obs, action, env_reward, next_obs,
                next_public, terminated):
        """Two terms, and they do deliberately different jobs.

        (1) POTENTIAL, F = gamma*Phi(s') - Phi(s) with

                Phi(s) = W_X * X_i(t,h)  +  W_B * B_i(e_i, rho_i),

        i.e. the potential is the value of the two terminal options the
        player currently holds: the exit value and the outlasting prize.

        Why this rather than a hand-written "reward for surviving": both
        terms are paid only at the very last stage of an episode, so plain
        PPO sees the credibility-flexibility trade-off of Sec. 4.5 only
        through one delayed scalar. Differencing them charges the *erosion*
        of the exit option (-lam*Delta K, humiliation, a closed mediation
        window) and the erosion of the prize (B falls as endurance is
        spent) at the stage where the choice is actually made.

        Being potential-based it telescopes over the episode and provably
        leaves the optimal policy unchanged (Ng, Harada & Russell 1999).

        (2) THE PRICE OF CONCEDING -- priced at zero, and the road there
        is the most useful thing we measured.

        The reasoning that put it in: conceding is the one act that raises
        the opponent's payoff directly, because they collect the whole
        outlasting prize B_j. If the ranking compares us against the field,
        handing that over should cost something, so we charged it

            -CONCEDE_PRICE * P(commitment type)
                           * E_posterior[B_j] / 175
                           * P(outlast the opponent)

        and three rounds of measurement went into those factors: a
        threshold detector just taught the policy to quit one stage before
        the detector could fire; an unconditional charge made it refuse to
        concede to *anyone*, costing 20 points of our own U.S. score in a
        larger field; and leaving out P(outlast) made it pay to stay in
        races it could not win -- worth -58.7 against an always-escalating
        opponent where taking the exit is worth about +20.

        The reasoning that took it out: all of it rests on the premise that
        the scoring rule is relative. It is not. Every reference agent on
        the leaderboard is scored against the reference pool, and a student
        submission is not a member of that pool, so our games never enter
        their averages. TitForTat's published score is a constant we cannot
        move. The criterion is therefore absolute:

            mean own utility over the reference pool  >  53.8

        against which any transfer term is pure cost with no return. Set to
        zero it also restores the property we wanted all along: the shaping
        is once again purely potential-based, so by Ng, Harada & Russell it
        provably leaves the optimal policy unchanged, and the policy is
        optimising exactly the quantity the board measures.

        The constant is kept rather than deleted because the term is correct
        for a genuinely relative rule -- the class round robin, where every
        submission does sit in every other submission's pool, is one, though
        there the weight is only 1/(N-1).
        """
        pid = player_id if player_id in ("I", "U") else "I"
        cur = self._potential(pid, obs)
        nxt = 0.0 if terminated else self._potential(pid, next_obs)
        f = GAMMA * nxt - cur

        if int(action) == ACT_EXIT and (CONCEDE_SEAT is None
                                        or pid == CONCEDE_SEAT):
            st = self._st.get(pid)
            if st is not None:
                f -= (CONCEDE_PRICE * st["p_commit"] * st["prize_j"]
                      * st["q_hat"])
        return float(f)

    @staticmethod
    def _potential(pid, obs):
        p = W[pid]
        o = np.asarray(obs, dtype=np.float64)
        rho_i, ehat_i = o[1], o[3]
        K_i, sig_j, med = 4.0 * o[4], o[7], o[8]
        X = p["x0"] - p["lam"] * K_i - p["h"] * sig_j + p["w_med"] * med
        B = p["b0"] * (p["b_base"] + p["b_e"] * min(ehat_i, 1.2)
                       + p["b_rho"] * rho_i)
        return W_X * X + W_B * B
