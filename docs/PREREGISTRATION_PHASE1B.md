# Phase 1b pre-registration: the same method with a frontier-only archive

**Frozen on 2026-11-26, before the first scored Phase 1b run.** Results go in
`docs/PHASE1B_RESULTS.md`.

## Question

Does the Phase 1 method pass the 20M Boulder gate once the archive stops
anchoring the policy to the opening area?

Phase 1 was stopped at 10.0M because 83% of resets loaded an archive pool full
of opening states. The policy did not collapse — entropy stayed at 1.83 — but
unique-map coverage fell from 7.1 to 3.6 and no curated event fired after the
opening. That run is not a fair test of recurrence or the learned tile map.
See [PHASE1_RESULTS.md](PHASE1_RESULTS.md).

## What changes from Phase 1

Only the archive policy. Recurrence, the learned 15x15 tile map, the reward,
gamma and the evaluation protocol are identical.

| setting | Phase 1 | Phase 1b |
|---|---:|---:|
| `checkpoint_weight` | 0.75 | **0.25** |
| `archive_frontier_fraction` | 0.50 | **0.80** |
| `archive_recent_fraction` | 0.30 | **0.20** |
| uniform opening-state draw | 0.20 | **0.00** |
| `archive_require_progress` | false | **true** |
| frontier-only save rule | none | skip if progress < pool best |

The frontier-only rule is the important one: a state is not saved if its
`badges*1000 + events` score is below the best score already in that worker's
pool. Opening-area states cannot refill the pool after the frontier moves.

## Fixed design

One training run, seed 43, 20M steps, 32 environments, from scratch.

- RecurrentPPO, shared LSTM 256, 32-step rollouts, batch 128.
- Observation: legacy seven + `local_tiles` (learned embedding + CNN).
- Reward: `explore_weight=3.0`, `level_weight=4.0`, `reward_scale=0.5`,
  gamma 0.995.
- Archive: per worker, max 50 states, save on new map/event/badge with a
  1,000-step cooldown, 25% of resets from the pool.
- HM automation off; agent off; no dynamic episode length.

## Protocol

- Checkpoints: 10M and 20M, fixed before the run.
- Evaluation: seeds 0, 1, 2, 61,440 steps, stochastic, cold `bulba_start`,
  no archive.
- Primary endpoint: **Boulder reach at 20M**.
- Secondary: Pewter, Viridian Forest, events, badges; archive pool size and
  reset percentage from training.

## Decision rules

1. **20M Boulder >= 2/3:** the archive fix works; extend the same run to 50M
   with the same method and evaluation. No re-tuning.
2. **20M Boulder < 2/3:** stop. The archive is not the whole explanation; the
   next phase is a method diagnosis, not another archive weight.
3. **Training-side badges are not results.** A `first_seen.csv` badge with
   0/3 evaluation is a transient.
4. **No reward or gamma tuning inside Phase 1b.** Only the archive settings
   above differ from Phase 1.

Seeds 10-19 remain reserved for the final confirmation and are not used for
any Phase 1b decision.

## What this run cannot establish

- It cannot separate the contribution of recurrence from the tile map; both
  are in the method. Phase 1 was the joint method test, and Phase 1b only
  fixes the archive.
- It cannot prove the archive is necessary; there is no no-archive arm at
  this budget. Phase 0's baseline is the no-archive reference at 20M.
- It cannot confirm the full game. The ceiling remains Mt. Moon until the
  policy passes Pewter reliably.

## Budget

20M at the measured ~560 steps/s is about 10 hours. The 10M checkpoint is
evaluated halfway, so a clear failure can be stopped early. Extension to 50M
only after a 20M pass.

## Artifacts

- Training: `runs/phase1b_recurrent_tiles_20M_phase1b_s43/`
- Checkpoints: `checkpoints/model_{10000000,20000000}.zip`
- Evaluations: `runs/phase1b_eval/<checkpoint>_model_<step>/`
- Results: `docs/PHASE1B_RESULTS.md`
