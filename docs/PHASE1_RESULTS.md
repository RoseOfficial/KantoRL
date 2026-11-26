# Phase 1 results: stopped early on archive anchoring

**Status:** stopped at 10.0M steps on 2026-11-26, before the pre-registered
20M gate. The pre-registered training diagnostics showed the archive was
anchoring the policy, so the gate would have confirmed a failure that was
already visible. This is **not** a gate result, and it is not a fair test of
recurrence or the learned tile map: 83% of resets came from the archive.

## Run

- Session: `runs/phase1_recurrent_tiles_50M_phase1_s43`
- Seed 43, 32 environments, 10.02M steps in 4h57m (~563 steps/s).
- Preserved checkpoint: `checkpoints/model_10000000.zip` (49 MB).
- Pre-registration: [PREREGISTRATION_PHASE1.md](PREREGISTRATION_PHASE1.md).

## What the diagnostics showed

| metric | opening | 10M |
|---|---:|---:|
| badges | 0 | 0 |
| curated events achieved | 0 | **4** (opening only) |
| events mean | 11 | 14.2 |
| unique maps mean | -- | 7.1 peak -> **3.6** |
| explore tiles mean | -- | 759 peak -> **298** |
| explore reward share | -- | **79%** |
| action entropy | 2.07 peak | 1.83 |
| archive pool | 0 | 58 states |
| checkpoint reset rate | 0% | **83%** |

The policy did not crash: action entropy stayed at 1.83 and the model kept
training at full speed. What collapsed was **progress**. Unique-map coverage
peaked at 7.1 and fell to 3.6; explore tiles peaked at 759 and fell to 298.
The only curated events are the four opening flags. At the same step the
Phase 0 baseline had already reached Viridian Forest.

The mechanism is visible in the reset rate: 83% of episode starts loaded an
archive state. The pool filled with opening-area states (11-14 events), and
the 75% reset weight kept sampling them. Each reset put the policy back near
the start, so it re-explored the same early maps instead of pushing the
frontier. This is the opposite of the intended retention effect.

## Why the early stop

- The 20M gate would answer "Boulder < 2/3", which the diagnostics already
  predict; waiting would burn ~5 more hours.
- Commit headroom had fallen to 1.8 GB. The run was at risk of the same
  Windows paging-file crash as Phase 0, which would have left a gate watcher
  waiting for a checkpoint that never arrives.
- The 10M checkpoint is enough to show the archive design is wrong. The
  recurrence and tile encoder were not given a fair test, so this is not a
  negative result for them.

## What remains valid

- The recurrent model with the learned tile embedding trains and runs at
  ~560 steps/s on the reference CPU.
- No action-entropy collapse and no code crash.
- The concurrent archive pruning fix works: no `PermissionError` in this run.
- The failure is a **design/weighting** problem in the archive, not a failure
  of recurrence or of the tile observation.

## Phase 1b

The next run is a new pre-registration, not a mid-run edit:

1. `checkpoint_weight` 0.75 -> **0.25**. Most episodes must start fresh so the
   policy can reach new ground; the archive is a retention supplement.
2. Archive sampling 50/30/20 -> **80% frontier / 20% recent**. The early-state
   pool must not dominate.
3. **Frontier-only saves**: skip a state whose progress score
   (`badges*1000 + events`) is below the pool's best. The opening-area states
   cannot fill the pool again.
4. Keep the 20M gate and the same evaluation protocol, so the comparison is
   direct.

Details: [PREREGISTRATION_PHASE1B.md](PREREGISTRATION_PHASE1B.md).
