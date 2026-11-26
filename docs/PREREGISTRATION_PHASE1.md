# Phase 1 pre-registration: recurrence, learned local map, self-generated archive

**Frozen on 2026-11-26, before the first scored Phase 1 run.** Results go in
`docs/PHASE1_RESULTS.md`. This file is not edited after the run starts.

A 100k-step smoke run of the exact tier completed before this freeze. It
verified the recurrent model, the learned tile encoder, the per-worker archive
and the archive save cooldown. It is not a scored result.

## Question

Does the method change produce and retain Boulder from scratch where the Phase
0 single-knob arms failed, and does it reach Pewter reliably, within a 50M-step
budget?

## Inputs from Phase 0

Phase 0 is in [PHASE0_RESULTS.md](PHASE0_RESULTS.md). The design uses four
findings:

1. **No single knob reached Boulder.** All four arms were 0/3 at 20M.
2. **Exploration moved the frontier.** `explore3` was the only arm to reach
   Pewter (2/3) and it removed the level-grinding signature (level share 0.02,
   explore share 0.90). `explore_weight=3.0` is carried forward.
3. **Gamma 0.999 did not transfer from scratch.** It was the weakest arm at
   20M. `gamma=0.995` is carried forward.
4. **Retention, not only exploration, is missing.** `level1` had the best 10M
   checkpoint and regressed by 20M; baseline and gamma lost training-side
   badges. Phase 1 selects checkpoints on validation and adds the archive.

## Design

One training run, seed 43, 50M steps, 32 environments.

| component | setting |
|---|---|
| policy | RecurrentPPO, shared LSTM 256, 32-step rollouts, batch 128 |
| observation | legacy seven + `local_tiles` (15x15) |
| local tiles | learned 256-entry embedding, two 3x3 convs, projected to 64 |
| reward | `explore_weight=3.0`, `level_weight=4.0`, `reward_scale=0.5`, gamma 0.995 |
| optimization | LR 1e-4, `target_kl=0.02`, 3 epochs, 32 envs |
| horizon | 61,440 steps, no dynamic extension |
| archive | per-worker pool, 75% of resets from the pool, max 50 states |
| archive saves | new map, badge, or event milestone; 1,000-step cooldown |
| archive sampling | 50% frontier, 30% recent, 20% uniform |
| HM automation | **off** |
| agent module | off |

Why these choices:

- **Recurrence** addresses room-to-room memory, menus and multi-step quests.
  A feedforward policy cannot represent "I came from the left" or "I already
  tried this door."
- **Learned local tile map** is the minimal-direction substitute for a
  hand-authored walkability map. The repository has no per-tileset collision
  table, and raw tile IDs are categorical; an embedding lets the model learn
  which tiles behave alike. No route, warp or sprite knowledge is added.
- **Archive** is the retention mechanism. It saves states the agent discovers
  itself and resets from them. It is training scaffolding only; every
  evaluation is a cold `bulba_start` with no archive.
- **No HM automation.** Scripted Cut/Surf is direction. The archive and the
  event reward are the generic substitute.

## Budget and stages

Measured throughput for this model on the reference CPU is 250-280 steps/s
from 100k-step runs. 50M is about 50-56 hours; 100M would be about 4.5 days.
Phase 1 is therefore staged:

| stage | steps | gate |
|---|---:|---|
| 1 | 20M | **stop if Boulder < 2/3** |
| 2 | 30M | **stop if Boulder < 2/3 or Pewter < 3/3** |
| 3 | 50M | success if Boulder >= 2/3 and Pewter 3/3 |

A failure at the 20M gate costs about 22 hours, not five days. A pass at 50M
is the precondition for extending to 100M. No stage is skipped.

## Evaluation protocol

- Checkpoints: 10M, 20M, 30M, 40M, 50M, fixed before the run.
- Seeds: 0, 1, 2 (validation). Stochastic policy, 61,440 steps, same
  `bulba_start` state, no archive, cold start.
- Primary endpoint: Boulder reach at 20M and 50M.
- Secondary: Pewter, Viridian Forest, events, badges, Route 3, Mt. Moon.
- Training diagnostics: the same reward shares, battle fraction, action
  entropy and map coverage added for Phase 0, plus archive pool size and
  checkpoint-reset percentage.
- No checkpoint sweep. The headline is the 50M checkpoint. If 50M regresses
  below the 30M gate, report the 30M checkpoint and the regression explicitly.

Seeds 10-19 remain reserved for the final fixed-ruleset confirmation and are
not used for any Phase 1 decision.

## Decision rules

1. **20M gate:** Boulder 2/3 or 3/3 -> continue to 30M. Otherwise stop and
   diagnose. Do not extend the run.
2. **30M gate:** Boulder >= 2/3 and Pewter 3/3 -> continue to 50M. Otherwise
   stop and diagnose.
3. **50M success:** Boulder >= 2/3 and Pewter 3/3 -> Phase 1 passes; extend to
   100M and then evaluate seeds 10-19.
4. **50M partial:** Boulder >= 2/3 but Pewter < 3/3 -> record as a partial
   result; do not extend without a new pre-registration.
5. **Training-side badges are not results.** The Phase 0 rule still applies: a
   badge in `first_seen.csv` with 0/3 evaluation is a transient.
6. **No reward or gamma tuning inside Phase 1.** A failed gate ends the phase,
   not triggers a fifth arm.

## Threats to validity

- One training seed. Gates are directional; the held-out confirmation is a
  separate, later step.
- The local tile map is learned, not a semantic walkability map. It is a
  minimal-direction approximation, not the full CJBoey minimap.
- The archive saves map-entry and milestone states, not every dungeon cell.
  A deep room with no event may not be in the pool.
- Throughput is measured on one machine and can fall when the machine is busy.
- The archive is only a training scaffold; a policy can still overfit to the
  states it samples. Cold-start evaluation is the guard.

## Artifacts

- Training: `runs/phase1_recurrent_tiles_50M_phase1_s43/`
- Checkpoints: `checkpoints/model_{10000000,20000000,30000000,40000000,50000000}.zip`
- Evaluations: `runs/phase1_eval/<checkpoint>_model_<step>/`
- Results: `docs/PHASE1_RESULTS.md`
