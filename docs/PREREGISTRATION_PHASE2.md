# Phase 2 pre-registration: isolate the archive and tile map with PPO

**Frozen on 2026-11-26, before the first scored Phase 2 run.** Results go in
`docs/PHASE2_RESULTS.md`.

## Question

Do the self-generated archive and the learned local tile map help the
feedforward PPO baseline, before any recurrence is added?

Phase 1 stacked three changes (recurrence, tile map, archive) and failed in two
different ways: 75% archive resets anchored the policy, and 25% with a
one-state pool removed the opening curriculum. Neither run isolates a
component. The recurrent model also runs at ~500 steps/s, so each 20M test
costs ~11h; PPO runs at ~1,800 steps/s, so the same test costs ~3h.

## Design

Four arms, seed 43, 20M steps, 32 environments, feedforward PPO. Everything
except the listed change matches `phase2_control`.

| arm | observation | archive | question |
|---|---|---|---|
| `phase2_control` | legacy seven | off | reference |
| `phase2_archive50` | legacy seven | 50% resets | does the archive help? |
| `phase2_tiles` | legacy + `local_tiles` | off | does the tile map help? |
| `phase2_tiles_archive50` | legacy + `local_tiles` | 50% resets | do they compose? |

Fixed settings: `explore_weight=3.0`, `level_weight=4.0`,
`reward_scale=0.5`, `gamma=0.995`, `learning_rate=1e-4`, `target_kl=0.02`,
61,440-step horizon, 3-frame stack, 72x80 screens, 24 frames per action.

The archive in both archive arms:

- 50% of resets from the pool, 50% fresh.
- Diverse map/event states; `archive_require_progress=false`.
- 50% frontier / 30% recent / 20% uniform sampling.
- Per-worker pool, max 50 states, save on new map/event/badge with a
  1,000-step cooldown.
- No HM automation, no dynamic episode length.

The control is deliberately the same reward as the arms, not the original
Phase 0 baseline. Phase 0's `explore3` arm used these settings and reached
Pewter 2/3 at 20M; it is an external sanity check, not the control.

## Why PPO first

The archive and tile map are independent of recurrence. Testing them on the
policy that is known to learn the early game avoids confounding a component
failure with a recurrent-training failure. Recurrence is added back only after
a component shows a gain.

## Evaluation

- Checkpoints: 10M and 20M, fixed before the run.
- Seeds: 0, 1, 2 (validation); 61,440 steps; stochastic; cold `bulba_start`;
  no archive.
- Primary endpoint: **Boulder reach at 20M**.
- Secondary: Pewter, Viridian Forest, events, badges, unique maps from
  training.
- No checkpoint sweep. Evaluate the fixed checkpoints.

## Decision rules

1. **Any arm reaches Boulder 2/3 or 3/3 at 20M:** that component helps. Carry
   it into the next method run and add recurrence only after a second
   confirmation.
2. **Boulder 0/3 but Pewter 2/3 or better:** the component improves navigation
   without the gym fight. Carry it, but the next experiment targets the
   battle/reward side, not the map.
3. **No arm beats the control:** the archive and tile map are not the
   bottleneck at this budget. Stop adding priors and diagnose training
   stability or the reward.
4. **No mid-run tuning.** A failed arm is a result, not a reason to change its
   settings inside Phase 2.

Seeds 10-19 remain reserved for the final confirmation.

## Budget

Four 20M arms at ~1,800 steps/s = ~3h each, ~12h total, plus four evaluations
at 10M/20M on three seeds (~20 minutes). The matrix runs sequentially to avoid
memory contention.

## Artifacts

- Training: `runs/phase2_<arm>_20M_phase2_s43/`
- Evaluation: `runs/phase2_eval/<session>_model_<step>/milestones.json`
- Results: `docs/PHASE2_RESULTS.md`
