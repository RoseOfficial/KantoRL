# Phase 0 results: no single knob reaches Boulder

**Status:** complete, 2026-11-26. Pre-registration:
[PREREGISTRATION_PHASE0.md](PREREGISTRATION_PHASE0.md). Four arms, seed 43,
20M steps each, evaluated at the fixed 10M and 20M checkpoints on validation
seeds 0-2 (three episodes per cell, 61,440 steps, stochastic, same
`bulba_start` state).

## Primary endpoint

| arm | 10M Boulder | 10M Forest | 10M Pewter | 20M Boulder | 20M Forest | 20M Pewter | training badge |
|---|---:|---:|---:|---:|---:|---:|---:|
| `phase0_baseline` | 0/3 | 0/3 | 0/3 | **0/3** | 3/3 (6,771) | 0/3 | 5.33M |
| `phase0_gamma999` | 0/3 | 0/3 | 0/3 | **0/3** | 0/3 | 0/3 | 13.62M |
| `phase0_explore3` | 0/3 | 0/3 | 0/3 | **0/3** | 3/3 (4,273) | **2/3 (32,120)** | none |
| `phase0_level1` | 0/3 | 3/3 (1,421) | 0/3 | **0/3** | 0/3 | 0/3 | 11.43M |

Medians are environment steps. All evaluation episodes ended with zero badges
and 11.7-15.3 event flags (the start state already has 11).

## Decision

**Pre-registration rule 2 applies: all four arms are 0/3 on Boulder at 20M.**
No single knob is sufficient. Phase 1 proceeds with the method change
(recurrence, semantic local map, intrinsic exploration, self-generated
archive) and stops tuning reward weights for the first badge. No fifth arm is
added, and no arm is extended past 20M.

Rule 4 also applies to the training-side badges: baseline, gamma and level1
each earned a Boulder Badge during training and each evaluated 0/3. They are
transients, not results.

## What the secondary endpoints show

**Stronger exploration moved the frontier, but not through the gym.**
`explore3` is the only arm to reach Pewter City from a cold start (2/3, median
32,120), and it did so with almost no level reward: final level share 0.02,
explore share 0.90, battle fraction 0.16. It never earned a badge even in
training. The reading is that coordinate novelty was too cheap in the
baseline, but exploration alone does not produce the Brock fight. It is a
carry-over candidate for Phase 1, not a Phase 0 win.

**Lowering the level weight helped early, then regressed.**
`level1` had the best 10M checkpoint: Viridian Forest 3/3 at a 1,421 median,
15.3 events, max level 15.3. By 20M it had fallen to Forest 0/3, 13.7 events,
max level 13.7. That is forgetting/retention, not a reward-balance problem.

**Gamma 0.999 did not transfer from scratch.**
The 104M warm-start probe found gamma 0.999 faster and more reliable on the
badge. From scratch at 20M it is the weakest arm: Forest 0/3, 11.7 events,
max level 6.0. The credit-horizon change alone does not fix the cold-start
run at this budget.

**The control reproduced the clean-run pattern.**
`baseline` reached Viridian Forest 3/3 at 20M, never Pewter, and its 5.33M
training badge evaluated 0/3. That is the same transient the clean 100M run
produced, so the screen is measuring the failure it was designed to measure.

## Final-window diagnostics

Shares are of positive reward in the last logging window; `entropy` is the
natural-log action entropy (maximum 2.079).

| arm | level share | explore share | battle fraction | unique maps | entropy |
|---|---:|---:|---:|---:|---:|
| `phase0_baseline` | 0.37 | 0.25 | 0.16 | 13.8 | 1.96 |
| `phase0_gamma999` | 0.58 | 0.42 | 0.35 | 10.0 | 1.96 |
| `phase0_explore3` | 0.02 | 0.90 | 0.16 | 11.6 | 1.89 |
| `phase0_level1` | 0.08 | 0.76 | 0.10 | 11.7 | 1.93 |

Two things are clear from the instrumentation:

1. **No arm collapsed.** Action entropy stayed between 1.89 and 1.96 in every
   arm, so the 20M failures are not a policy collapse in the action
   distribution.
2. **The level mix is not the whole story.** The arm with the highest level
   share (gamma) was the weakest; the arm with the lowest (explore3) reached
   furthest. But `level1` also had a low level share and still regressed
   between 10M and 20M. Exploration explains part of the plateau; retention
   explains the rest.

## What Phase 1 should change

The screen rules out another reward-weight arm. It does not rule out the
method change, and it gives three concrete inputs:

1. **Keep gamma 0.995 for the first method run.** Do not carry 0.999; it was
   negative from scratch at 20M.
2. **Carry `explore_weight=3.0`.** It is the only setting that reached
   Pewter, and it removed the level-grinding signature. Keep `level_weight`
   at 4.0; the low level share in explore3 came from not fighting, not from
   the weight being wrong.
3. **Select checkpoints by validation, not by final step.** `level1` at 10M
   is better than at 20M, and the baseline's training badge is a transient.
   Phase 1 should evaluate every ~5M and keep the best by a fixed gate.

The Phase 1 method is the one proposed before the screen: RecurrentPPO
(LSTM), a tile-level semantic local map, RND/count-based intrinsic
exploration, and the self-generated checkpoint archive. The archive is the
part that addresses the 10M-to-20M regressions directly.

## Threats to validity

- Three evaluation episodes per cell and one training seed. This is a screen,
  not a confirmation; 2/3 and 3/3 are directional.
- The 20M budget is below the clean run's 24.1M stop, and far below the
  ~800M reference. A knob that fails here could still help at scale.
- `explore3` was rerun after the first attempt crashed on a Windows commit
  limit. The rerun used the identical frozen config and produced all its
  data; the crashed attempt produced none.
- The fixed 10M/20M checkpoints are not a sweep. A better policy may exist
  between them, which is why Phase 1 will select on validation.
- Training-side `first_seen.csv` is not an endpoint. Three arms earned a
  training badge and none retained it.

## Artifacts

- `runs/phase0_baseline_20M_phase0_s43/`
- `runs/phase0_gamma999_20M_phase0_s43/`
- `runs/phase0_explore3_20M_phase0_s43/`
- `runs/phase0_level1_20M_phase0_s43/`
- `runs/phase0_eval/*/milestones.json`
- `runs/phase0_sweep.log` (first pass, includes the crash)
- `runs/phase0_sweep2.log` (explore3 rerun and all evaluations)
