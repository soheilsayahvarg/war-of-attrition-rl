# Strategic Endurance under Uncertainty

A reinforcement-learning agent for a **multi-stage war of attrition with
incomplete information**, built for the Game Theory course at Sharif
University of Technology (CE, Spring 2025–26).

The agent reached **1st among students** on the class practice leaderboard,
above both reference marks the rubric sets:

| | Score | as Iran | as U.S. |
|---|---|---|---|
| **This project's best board checkpoint** | **61.7** | 72.9 | 50.4 |
| Instructor benchmark (`MasterAgent`) | 56.3 | 65.7 | 46.9 |
| `TitForTat` baseline | 53.8 | 69.7 | 37.9 |
| `Hawk` baseline | 51.5 | 71.9 | 31.2 |

That is not the checkpoint that ships here, and the reason is the most
interesting result in the project — see *The scoring rule was not what the
leaderboard shows* below.

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

**5. Report what failed.** Eight hypotheses were falsified by data and all
eight are in the report, including three I had already announced as working.
One section quantifies how much of the leading board score was seed luck:
that checkpoint sits **1.5σ above the mean of its own 14-checkpoint family**
(min 51.5, median 58.7, σ = 2.33), so a good part of 61.7 is a favourable
draw rather than method.

## The scoring rule was not what the leaderboard shows

The leaderboard scores against 5 scripted baselines plus the instructor
benchmark. The *graded* tournament is a round robin over every submission,
**2** reference agents (`Random`, `Tit-for-Tat`) and the benchmark — Hawk,
Dove and BayesianThreshold are practice-only. Those are different objectives,
and on this problem they disagree sharply: a mirroring opponent is weak
against scripted pushovers and strong against RL agents that have learned to
escalate, so `TitForTat` scores 53.8 on the board and ~84 in the tournament.

**The board leader loses the graded criterion.** The 61.7 checkpoint, first
among students, finishes *below* `TitForTat` in a simulated graded run.
Everything tuned against the board had been optimising a rank that carries
no marks.

Two ideas were tried on top of the corrected objective. Both worked on one
opponent set and died on another, which is the point:

- **Seat grafting** (`analysis/graft.py`). A checkpoint holds two role
  networks and the loader instantiates them independently, so the Iran seat
  and the U.S. seat can come from different runs — the compromise each
  training run makes between them was never required. The seat scores added
  *exactly*, taking the margin over `TitForTat` from +1.7 to +11.2.
- **Policy sharpening** (`analysis/sharpen.py`). The tournament constructs
  agents with `sample=True`, hard-coded, so a policy is never evaluated at
  its argmax. Scaling the actor head by τ is exactly a softmax temperature of
  1/τ: the ordering of legal actions, and therefore the maximising policy, is
  unchanged — only the noise around it shrinks.

**What settled it was the leaderboard itself, not a better simulation.** The
board publishes *both seat scores* for every student who has uploaded, so the
graded field does not have to be guessed: each classmate is stood in for by
whichever checkpoint lands nearest their published two-seat profile
(`analysis/classfield.py`). Against that field every classmate turns out to
be Iran-strong and U.S.-weak, the grafted checkpoint's +11.2 collapses to
−7.7, and sharpening flips from noise (+2.5 / −1.1 across two seeds) to a
real +2.3 on all three.

## Results

Board score across four uploads, with the correction that produced each:

| Upload | Change | Score |
|---|---|---|
| 1 | tuned on the local 2-opponent harness | 52.9 |
| 2 | trained on the reconstructed 6-opponent pool; denial term removed | 56.8 |
| 3 | patient archetypes restored to the pool | 58.8 |
| 4 | opponent-behaviour features (19 total) | **61.7** |

Every number above is measured on the real leaderboard, not estimated.

The shipped checkpoint is a different one, chosen on the graded criterion.
Four candidates in one field, mean of three seeds:

| Checkpoint | Board | Margin over `TitForTat` | Rank |
|---|---|---|---|
| board leader (61.7) | **61.7** | −1.9 | 5–6 |
| `gr_k` (seat graft) | 61.1 | +3.6 | 3 |
| `gr_l` (seat graft) | 60.2 | +4.8 | 2 |
| **shipped** (`of_s3`, τ = 1.5) | 59.4 | **+8.0** | **1 on all three seeds** |

The trade is regular: each 0.8 board points buys about 2 tournament points.
The board carries no marks and the tournament carries half the implementation
grade, so the shipped checkpoint gives up the former — while still sitting
5.6 points above `TitForTat` on the board, so the "beat Tit-for-Tat"
criterion holds under either reading of where it is measured.

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
