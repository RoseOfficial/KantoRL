# Phase 0 pre-registration: why did the clean from-scratch run fail?

**Frozen on 2026-11-25, before the first scored Phase 0 run.** Results go in
`docs/PHASE0_RESULTS.md`. This file is not edited after the runs start; if the
design turns out to be wrong, that is recorded as a result, not repaired in
place.

A 53k-step smoke run (`runs/phase0_smoke`, deleted) verified that the new
reward/action instrumentation writes to TensorBoard before this freeze. It is
not one of the four arms, produced no evaluation report, and was not used to
choose any setting.

## Question

The clean from-scratch run stopped at 24.1M with 0/3 Boulder at 10M, 20M and
24.1M. Its 5.33M first badge was training-only. Four candidate causes remain:

1. **Credit horizon** — `gamma=0.995` at 24 frames per action is a ~200-decision
   horizon, while Brock is ~7,000 steps away.
2. **Novelty weight** — `+0.015` per new coordinate is small next to `+2.0` per
   level, so walking to a new town pays less than one wild battle.
3. **Level attractor** — the max-level potential is the largest early reward
   term and can be farmed in tall grass, which is where the clean run stalled.
4. **Method limit** — a feedforward PPO policy starting from one state cannot
   represent or revisit progress, so no single knob is sufficient.

Phase 0 is a screen: it asks which of the first three axes, if any, moves the
from-scratch result. It is not a confirmation run.

## Prior evidence this design uses

- The 99M lineage is path-dependent; the clean run of the same rules failed.
- A 104M warm-start probe found `gamma=0.999` faster on the badge (5,160 vs
  7,182 median) and more reliable on Mt. Moon 3 Trainer 2, at the cost of Mt.
  Moon 1 depth.
- The reward-rebalance and local-prior warm starts all regressed. Phase 0 does
  not repeat those warm starts; it changes one axis from scratch.
- At 20M the clean run reached Viridian Forest 3/3 but Pewter 0/3.

## Fixed design

Four arms, one training seed, 20M steps. Everything except the listed axis is
identical to `phase0_baseline`.

| arm | tier | only change | hypothesis |
|---|---|---|---|
| control | `phase0_baseline` | none | reproduces the clean-run failure |
| horizon | `phase0_gamma999` | `gamma` 0.995 -> 0.999 | credit assignment is the blocker |
| novelty | `phase0_explore3` | `explore_weight` 0.3 -> 3.0 | coordinate novelty is too cheap |
| level | `phase0_level1` | `level_weight` 4.0 -> 1.0 | the level term crowds out progress |

Held constant: seven-component legacy observation, 3-frame stack, 72x80
screens, `action_freq=24`, `max_steps=61,440`, `stagnation_steps=30,720`,
`reward_scale=0.5`, `target_kl=0.02`, 32 environments, `learning_rate=1e-4`,
seed 43, no curriculum, no agent module.

Why one seed: this is a screen, not a confirmation. A positive arm is carried
into Phase 1 and replicated there. Seeds 10-19 remain reserved and are not
touched in Phase 0.

## Evaluation protocol

- Checkpoints: `model_10000000.zip` and `model_20000000.zip` from each run.
- Evaluation seeds: 0, 1, 2 (validation seeds; not used for training).
- Budget: 61,440 steps, stochastic policy, same `bulba_start` save state.
- No checkpoint sweep. The 10M and 20M checkpoints are fixed before the run;
  the `best_model` snapshot is not used for the primary endpoint.
- The training-side `first_seen.csv` is reported separately and is not a
  substitute for evaluation.

Command:

```powershell
python scripts/run_phase0.py "C:/roms/Pokemon Red.gb" `
    --save-state "C:/roms/states/bulba_start.state"
```

## Primary endpoint

**Boulder Badge reach at 20M, 3 evaluation episodes per arm.**

- A positive signal requires **2/3 or 3/3**.
- 1/3 is inconclusive.
- 0/3 is negative.

## Secondary endpoints (descriptive)

- Boulder reach and median at 10M.
- Viridian Forest and Pewter City reach at 10M and 20M.
- `badges` and `events` per episode from the milestone report.
- Training diagnostics, from `TensorBoard` under `game/`:
  - `reward_level_share`, `reward_event_share`, `reward_explore_share`
  - `battle_fraction`
  - `action_entropy`
  - `unique_maps_mean`
- First badge step from `first_seen.csv`, if any.

Definitions:

- Reward shares are each component's sum divided by the sum of all positive
  components over the logging window. The stuck penalty is excluded from the
  denominator.
- `battle_fraction` is the fraction of environment steps with `in_battle=True`.
- `action_entropy` is the natural-log entropy of the action histogram over the
  window; the maximum is `ln 8 = 2.079`.
- `unique_maps_mean` is the mean number of distinct map IDs visited per episode
  in the window.

## Decision rules

These rules are fixed. They are not thresholds to revisit after seeing results.

1. **Any arm at 2/3 or 3/3 Boulder at 20M** — that axis is implicated. Phase 1
   carries the winning setting into the recurrence + local-map method. No fifth
   arm is added in Phase 0.
2. **All arms 0/3 at 20M** — no single knob is sufficient. Phase 1 proceeds with
   the method change (recurrence, semantic local map, intrinsic exploration,
   archive) and stops tuning reward weights for the first badge.
3. **An arm at exactly 1/3** — inconclusive. Record it and move on; do not
   extend the arm or retune it inside Phase 0.
4. **A badge in `first_seen.csv` but 0/3 at evaluation** — transient, not a win.
   This is the failure mode of the 5.33M badge in the clean run.
5. **No arm is extended past 20M.** A follow-up requires a new pre-registration.
6. **No arm is rerun with different settings inside Phase 0.** A rerun of the
   same arm is allowed only to replace a crashed run, with the same config.

## Diagnostic expectations

These are interpretations, not gates.

| pattern | reading |
|---|---|
| level share > 50%, battle fraction > 50%, unique maps flat | the level attractor is the stall |
| action entropy < 0.5 nats | policy collapse, not exploration failure |
| explore share rises but badges do not | wandering without progress |
| gamma arm wins | credit horizon is the blocker |
| level arm wins | reward mix is the blocker |
| novelty arm wins | coordinate novelty is the blocker |
| all arms fail | method limit; single knobs cannot fix it |

A single diagnostic can be misleading; the decision uses the primary endpoint
plus the reward mix and action entropy together.

## Budget

Four runs x 20M = 80M environment steps. At the measured ~1,800 steps/s that
is about 12.3 hours, plus four arms x two checkpoints x three episodes x 61,440
steps of evaluation (about 1.5M steps, under 20 minutes of harness time plus
emulator startup). One attempt.

## Artifacts

- Training: `runs/phase0_<arm>_20M_phase0_s43/`
- Evaluation: `runs/phase0_eval/<session>_model_<step>/milestones.json`
- Results write-up: `docs/PHASE0_RESULTS.md`
