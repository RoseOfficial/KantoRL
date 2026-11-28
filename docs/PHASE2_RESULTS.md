# Phase 2 results: the tile map helps, then collapses

**Status:** complete, 2026-11-27. Four arms, 20M steps each, seed 43,
evaluated at the fixed 10M and 20M checkpoints on validation seeds 0-2.
Pre-registration: [PREREGISTRATION_PHASE2.md](PREREGISTRATION_PHASE2.md).

## Primary endpoint: Boulder at 20M

| arm | Boulder | Forest | Pewter | events | max level |
|---|---:|---:|---:|---:|---:|
| control | **0/3** | 3/3 | 2/3 | 14.0 | 10.0 |
| archive50 | **0/3** | 2/3 | 2/3 | 13.3 | 8.3 |
| tiles | **0/3** | 0/3 | 0/3 | 11.0 | 6.0 |
| tiles + archive50 | **0/3** | 0/3 | 0/3 | 13.7 | 6.0 |

No arm reached Boulder at 20M. The control reproduces the Phase 0
`explore3` result exactly (0/3 Boulder, Pewter 2/3, Forest 3/3), so the
matrix is measuring what it was meant to measure.

## The 10M tiles checkpoint is the real result

| arm at 10M | Forest | Pewter | events | max level | reward |
|---|---:|---:|---:|---:|---:|
| control | 0/3 | 0/3 | 14.0 | 6.0 | 33.7 |
| archive50 | 3/3 (14,030) | 0/3 | 13.3 | 7.0 | 83.9 |
| **tiles** | **3/3 (1,302)** | **3/3 (23,113)** | **17.3** | **16.0** | **182.6** |
| tiles + archive50 | 1/3 | 1/3 | 14.7 | 7.0 | 60.2 |

The tile map is the first component in the project to clearly improve the
from-scratch early game: all three seeds reach Pewter, Viridian Forest is
cleared at a 1,302 median, and the party reaches level 16. The control at 10M
is still at level 6 and has not reached Pewter.

Then the 20M checkpoint collapses all the way back to the start:

| tiles | 10M | 20M |
|---|---:|---:|
| Forest | 3/3 | 0/3 |
| Pewter | 3/3 | 0/3 |
| events | 17.3 | **11.0** (start value) |
| max level | 16.0 | 6.0 |
| explore tiles | 66 | **7** |
| reward | 182.6 | **3.1** |

The training-side policy at 20M still has 14.97 events and 11.6 maps, so this
is not a failure to explore in training. It is a **late-training drift** that
destroys the cold-start policy: the 20M checkpoint does not reproduce the
10M behaviour.

## Archive result

The archive did not help. At 20M it is slightly behind the control on Forest
(2/3 vs 3/3) and Pewter is the same (2/3); training-side it is slower to reach
Pewter (18.9M vs 15.4M). Adding it to the tiles arm made the 10M checkpoint
worse (1/3 Pewter instead of 3/3). The archive is not the missing retention
mechanism.

## Decision

**Rule 2 applies: Boulder 0/3, but Pewter 2/3+.** The tile map is a navigation
gain, not a badge. The next experiment targets the battle/retention side, not
more observation priors.

Three facts define Phase 3:

1. The tile map is the component to carry forward. At 10M it is strictly
   better than the control on every early-game metric.
2. The 20M collapse is a retention problem, not an exploration problem. The
   training-side metrics stay healthy while the cold-start checkpoint dies.
3. The archive is not the fix. It is dropped from Phase 3.

Phase 3 should be the tiles arm with a stability change: a lower late learning
rate (1e-4 -> 5e-5 at 10M) and fixed evaluation every 2.5M (10, 12.5, 15,
17.5, 20M) so the peak and the collapse are both measured. The gate is a
Boulder 2/3 at any pre-registered checkpoint, confirmed on held-out seeds.

## Threats

- Three evaluation episodes per cell, one training seed. The 10M tiles gain is
  strong (3/3 vs 0/3) but still needs replication.
- The 10M checkpoint is secondary to the pre-registered 20M primary endpoint.
  The collapse is the primary result; the 10M gain is the hypothesis for Phase 3.
- The 10M tiles policy reaches Pewter at level 16 but still does not beat
  Brock. Navigation and the gym fight are separate problems.
- The archive arms had a high reset rate and a shared-pool history; the
  per-worker pools were used, but the archive conclusions apply to this
  implementation only.

## Artifacts

- `runs/phase2_control_20M_phase2_s43/`
- `runs/phase2_archive50_20M_phase2_s43/`
- `runs/phase2_tiles_20M_phase2_s43/`
- `runs/phase2_tiles_archive50_20M_phase2_s43/`
- `runs/phase2_eval/*/milestones.json`
- `runs/phase2_sweep.log` and `runs/phase2_sweep_resume.log`
