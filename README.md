# Strategic Endurance under Uncertainty

A reinforcement-learning agent for a **multi-stage war of attrition with
incomplete information**, built for the Game Theory course at Sharif
University of Technology (CE, Spring 2025–26).

The agent finished **1st among students** on the class leaderboard, scoring
above both reference marks the rubric sets:

| | Score | as Iran | as U.S. |
|---|---|---|---|
| **This agent** | **61.7** | 72.9 | 50.4 |
| Instructor benchmark (`MasterAgent`) | 56.3 | 65.7 | 46.9 |
| `TitForTat` baseline | 53.8 | 69.7 | 37.9 |
| `Hawk` baseline | 51.5 | 71.9 | 31.2 |

Score is mean raw utility against the reference pool, averaged over both
roles on a common seed list.

---

## The game

Two players with privately known *structural types* θ = (ρ̄, m̄) burn a
latent endurance stock over up to 40 stages. Each stage a player picks a
public signal σ ∈ {0, ¼, ½, ¾, 1} or exits. Signalling is costly, costs
compound as endurance depletes, and public commitment K erodes the value of
the outside option — so the decision to keep fighting is a genuine
continuation gate, not a fixed threshold.

The interesting structure: endurance is **latent**, but the law that moves
it is **public**. Conditional on a hypothesised type θⱼ, the opponent's
entire endurance trajectory is pinned down.

## What I built

**A Bayesian filter over the opponent's type, not their trajectory.** The
above observation collapses trajectory inference to a posterior over a
2-D type, carried on a 11×9 grid and updated from four channels:

- **Survival** — an opponent still in the game rules out every type whose
  simulated endurance would already have hit zero. Exact, and needs no
  model of their strategy.
- **Costly signalling** — single-crossing (∂²φ/∂σ∂e < 0) makes a sustained
  loud posture evidence of a strong type, via a quantal-response likelihood.
- **The published asymmetric prior.**
- **Commitment types** — a fixed reply rule σⱼ(t) = σᵢ(t−1) is invisible to
  the three channels above, because it is independent of θⱼ. It is
  identified by the rule itself, as an explicit posterior rather than a
  counter with a threshold.

From that posterior come closed-form decision statistics: a collapse time
T = e₀·m̄·(0.35x + 0.325x²)/R, the probability of winning the attrition
race, the continuation gate, the erosion rate of the exit option, and the
value of the mediation option.

**Potential-based reward shaping.** Training reward is shaped strictly as
Φ(s') − Φ(s), so by Ng, Harada & Russell (1999) the optimal policy is
unchanged — the agent maximises exactly the quantity the tournament scores.
An earlier version carried an extra "deny the opponent the prize" term;
Section 11 of the report shows why that was pure cost under this scoring
rule and removing it was worth ~14 points.

**Behavioural features.** The 16 posterior features all describe what the
opponent *is*. Three raw statistics of what it *does* (signal standard
deviation, running max, fraction of stages silent) were added after
measurement showed the agent could not separate an erratic opponent from a
patient one — they are different *rules*, not different *types*.

## The part I would actually put on a CV

Most of the work was not the model. It was finding out that **I was
optimising the wrong objective, three times**, and building the measurement
apparatus to notice.

**1. The local harness was not the scoring rule.** `scripts/evaluate.py`
scores against 2 opponents; the leaderboard scores against 6. The two move
in *opposite* directions here — a checkpoint reading +10.7 locally scored
−0.9 on the board.

**2. So I reverse-engineered the leaderboard.** The three missing reference
agents were reconstructed from a design document and fitted to published
aggregate rows (`my_team/reference_pool.py`, `analysis/board.py`). Fitting
also revealed an unwritten rule: each reference agent's score **includes its
own mirror match**, which is why an always-escalate Hawk ranks third rather
than first. Four of the six references then reproduced the board **to the
decimal, on all four independent uploads** — and the instructor later
released the harness module, which turned out byte-identical to the copy
this project had been running (`analysis/verify_tournament.py`).

**3. When the reconstruction failed, I stopped fitting it.** Two references
never reproduced under any parameter setting. Rather than tune harder, I
looked for a *proxy that preserves the ordering* (`analysis/proxy_probe.py`)
and rebuilt checkpoint selection to keep the two kinds of evidence apart —
exact slots ranked directly, unidentifiable ones only via a proxy, never
averaged into one number (`analysis/select2.py`). The estimator reports how
far outside its calibration range each candidate sits and **refuses to
extrapolate** beyond it.

**4. Diagnose behaviour, not scores.** The costliest bug — a policy that
pinned σ = 0.75 for 96% of stages and never took the exit — was invisible in
aggregate score and obvious the moment behaviour was measured against a
passive opponent (`analysis/sigprofile.py`, `analysis/seat_probe.py`).

**5. Report what failed.** Five hypotheses were falsified by data and all
five are in the report, including two I had already announced as working.
The final section quantifies how much of the winning score was seed luck:
the shipped checkpoint sits **1.5σ above the mean of its own 14-checkpoint
family** (min 51.5, median 58.7, σ = 2.33), so a good part of 61.7 is a
favourable draw rather than method. Twelve further checkpoints across five
distinct interventions failed to beat it.

## Results

Board score across four uploads, with the correction that produced each:

| Upload | Change | Score |
|---|---|---|
| 1 | tuned on the local 2-opponent harness | 52.9 |
| 2 | trained on the reconstructed 6-opponent pool; denial term removed | 56.8 |
| 3 | patient archetypes restored to the pool | 58.8 |
| 4 | opponent-behaviour features (19 total) | **61.7** |

Every number above is measured on the real leaderboard, not estimated.

## Layout

```
my_team/          the contribution: theory, opponent archetypes, training driver
  theory.py         Bayesian filter + potential-based shaping (submitted)
  reference_pool.py reconstruction of the leaderboard's reference agents
  opponents.py      training archetypes
  train_plus.py     training driver with configurable opponent pools
analysis/         ~30 single-purpose experiment scripts, one per question asked
docs/report.pdf   full report (40 pages, Persian)
docs/slides.pdf   defence slides (22 pages, Persian)
see/              course-provided engine — see NOTICE.md
```

## Reproducing

```bash
pip install -r requirements.txt

# train against the leaderboard's reference pool plus patient archetypes
python my_team/train_plus.py --theory my_team/theory.py --team <id> \
    --steps 700000 --pool board+p --entropy 0.02 --entropy-final 0.004 \
    --out runs/agent.pt

# rank candidates, keeping exact and proxy evidence separate
python analysis/select2.py runs/agent.pt

# per-opponent breakdown against the reconstructed board
python analysis/board_detail.py runs/agent.pt
```

Training a single 700k-step checkpoint takes roughly one hour on 2 CPU
threads.

## Attribution

The `see/`, `scripts/`, `tests/`, `benchmarks/` and `docs/DESIGN.md` trees
are course-provided material and are not my work. See [NOTICE.md](NOTICE.md).
