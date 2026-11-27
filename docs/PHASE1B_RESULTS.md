# Phase 1b results: stopped at 3.6M, the archive fix overcorrected

**Status:** stopped at 3.6M steps on 2026-11-26, before the 20M gate. A
diagnostic evaluation of the 3.5M checkpoint shows the policy never left the
opening area. This is not a clean test of the archive change: the recurrent
policy was not learning the early game at this budget.

## What happened

Phase 1b changed only the archive: 25% checkpoint resets (was 75%), 80/20
frontier/recent sampling, and frontier-only saves (skip a state below the
pool's best progress).

Training at 3.6M:

| metric | Phase 1 at 8.4M | Phase 1b at 3.6M |
|---|---:|---:|
| badges | 0 | 0 |
| events mean | 14.4 | **11.0** (start value) |
| unique maps mean | 5.7 | **1.08** |
| explore tiles mean | 582 | **25** |
| archive pool | 58 | **1** |
| checkpoint reset rate | 83% | **0%** |

Diagnostic evaluation of `model_3500000.zip`, seeds 0-2, 61,440 budget:

| seed | steps | events | tiles | Boulder |
|---|---:|---:|---:|---:|
| 0 | 10,251 | 11 | 6 | 0 |
| 1 | 10,252 | 11 | 6 | 0 |
| 2 | 10,241 | 11 | 1 | 0 |

Episodes ended after ~10k steps, not the 61,440 budget: the policy stalls and
the environment ends the episode. No seed left the opening map.

## Reading

The frontier-only rule left the pool with **one state** (the opening). With no
progress, there was no frontier state to save, and with 25% resets the policy
got little practice through the opening. Phase 1's 75% resets from early
states were doing curriculum work: they repeatedly placed the policy just
after the opening, which is how it reached 5.7 maps. The archive was both an
anchor and a curriculum; removing it entirely removed the curriculum too.

Neither extreme works:

- 75% early-state resets: anchors the policy, map coverage collapses by 10M.
- 25% resets with a one-state pool: no curriculum, policy stalls at the
  opening.

The deeper problem is that the **recurrent + tile-map policy is not learning
the early game at this budget**. At 3.6M steps it is still near the start, so
the archive cannot be evaluated cleanly. Phase 1 stacked three changes
(recurrence, tile map, archive) and the failure cannot be attributed.

## Next: isolate the components with PPO

The feedforward PPO baseline learns the early game and runs at ~1,800
steps/s, 3-4x faster than the recurrent model. The next phase is a cheap
20M-step ablation, one change per arm:

1. **Control:** legacy observation, no archive. Already known: Forest 3/3,
   Boulder 0/3 at 20M.
2. **Archive:** legacy observation + archive at 50% resets with diverse
   map/event states. Does the archive help the known baseline?
3. **Tile map:** legacy observation + `local_tiles` (learned embedding), no
   archive. Does the local map help the known baseline?
4. **Both:** legacy + tile map + archive, 50%.

At 1,800 steps/s each 20M arm is ~3 hours; the matrix is ~12 hours, cheaper
than one recurrent 20M run. Only if the archive or tile map shows a gain does
recurrence get added back.

The 3.6M checkpoint and logs are preserved. The pre-registration is in
[PREREGISTRATION_PHASE1B.md](PREREGISTRATION_PHASE1B.md).
