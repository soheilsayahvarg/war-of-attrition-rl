# Attribution

This repository mixes my own work with material provided by the course, and
the split matters. Nothing here should be read as a claim over the engine.

## Course-provided (not my work)

Authored by the course staff of the Game Theory course, Department of
Computer Engineering, Sharif University of Technology:

| Path | What it is |
|---|---|
| `see/` | the game engine, environment, PPO trainer and tournament runner |
| `scripts/` | the handed-out training and evaluation entry points |
| `tests/` | the environment test suite |
| `benchmarks/master_agent.pt` | the instructor's benchmark agent |
| `docs/DESIGN.md` | the engine design document |
| `requirements.txt` | pinned dependencies |

Not one byte under `see/` was modified. That was a hard requirement of the
assignment, and it is verifiable: in the original course repository,
`git status --porcelain see/` returns empty.

## Mine

| Path | What it is |
|---|---|
| `my_team/` | the agent: belief features, reward shaping, opponent archetypes, training driver |
| `analysis/` | every experiment, diagnostic and selection script |
| `docs/report.pdf`, `docs/slides.pdf`, `docs/report/` | the report and defence slides |
| `submission.pt`, `runs/` | trained checkpoints |

## Why this repository is private

It contains a graded submission for a course whose competitive component may
still be running for other students. It is not published.
