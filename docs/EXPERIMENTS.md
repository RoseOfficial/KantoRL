# Experiment log

Working notes for the observation-tier ablation. Every entry records what was
measured, not what was hoped. Numbers are from the reference machine
(8-core/16-thread desktop, PyBoy 2.9.0, `fast_mode=True`).

## Setup

- Pokémon Red (UE), 24 emulated frames per action, button held 8 frames, the
  final frame rendered (see bug 2).
- 32 parallel envs, `gamma=0.995`, PPO with 128 steps per env per update.
- Episodes start from `roms/states/bulba_start.state` — Oak's Lab, level-6
  Bulbasaur, 11 event flags already set. This matches what the frontier
  projects do and skips the prologue.
- The three tiers differ in **nothing** except `obs_components`; a test
  (`test_all_tiers_share_reward_and_horizon`) enforces that.

| tier | observation |
|---|---|
| `t0_pure` | `screens` (3, 72, 80) |
| `t1_minimal` | + `health` (1,) |
| `t2_pragmatic` | + `level` (8,), `badges` (8), `events` (2560), `map` (48, 48), `recent_actions` (3) |

## Harness bugs found

Each one masked the results of the runs before it. In order of discovery:

1. **Episode longer than the training budget.** `max_steps=163_840` while a
   2M-step run across 32 envs gives each env only ~62,600 steps. No episode
   ever ended, so novelty rewards never refreshed and the reward went flat
   after the first room. All three tiers trained policies *worse than uniform
   random* (starters 0/3, 1/3, 0/3 versus random 3/3). Fixed with a 20,480 cap
   plus a stagnation window (10,240 steps, 1.0 reward).

2. **Frozen screens.** PyBoy 2.9 does not update the screen buffer when
   `tick(render=False)`. Every screen observation in every earlier run was the
   same still frame: measured 0 changes in 120 steps. The pixel-only tier was
   blind. Fixed by rendering the final frame of each action; costs ~1%
   (301 vs 299 steps/s) and restores 119/120 updates.

3. **Initialisation flags reported as achievements.** A new game sets bits
   0/25/32/33/34/35/39/960 at once, before the player can act. Reported as
   "Got Poke Flute", "Beat Giovanni Silph Co" at step 8. Fixed by treating the
   first non-zero flag batch as a baseline.

4. **Reward paid for progress that already existed.** 28 of 50 reward per
   episode (56%) came from those same initialisation flags. Fixed by
   baselining the event and badge deltas at the first non-zero count.

5. **Interaction collapse.** Trained policies pressed A on 1.5% (T0) and 4.6%
   (T2) of steps versus 12.3% for uniform random, so they never picked up a
   starter. A is strictly bad under a movement reward: it opens dialogue,
   which freezes position and stops paying novelty. Fixed with an
   interaction-novelty reward and `stuck_penalty=0`.

6. **Evaluation used a different start state.** `kantorl milestones` built a
   fresh config with no save state, so policies trained in Oak's Lab were
   scored from a cold ROM boot. Fixed with `--save-state`.

7. **Event labels and indices were invented.** `events.json` mapped
   plausible-sounding late-game names onto arbitrary bits, and "Got Starter"
   watched bit 50 when the real bit is 34 — so it could never fire. All 507
   labels are now generated from `pret/pokered`
   `constants/event_constants.asm`, verified against a save state whose set
   bits decode to exactly the prologue. `EVENT_FLAGS_SIZE` widened 312 → 320
   bytes to match `NUM_EVENTS = 0xA00`.

## Throughput

`fast_mode=True` is 3.2x per env (299 → 956 steps/s single env).

Environment ceiling (`bench_throughput.py`, IPC included):

| envs | 4 | 8 | 16 | 24 | 32 | 48 |
|---|---:|---:|---:|---:|---:|---:|
| steps/s | 2,181 | 3,272 | 4,562 | 5,405 | 5,750 | 6,508 |

Training loop (`bench_train.py`, rollout + update):

| envs | 16 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|
| steps/s | 1,558 | 1,835 | 1,880 | 1,913 |

The update now takes 44-47% of wall time, so more envs buy little. A 20M run
is ~2.5-3 hours.

## Results

**2M steps, T0 from `bulba_start`:** Route 1 in 2/5 episodes, Viridian City
0/5, no badges.

**20M steps, 3 training seeds x 5 evaluation episodes each (15 episodes per
tier), all from `bulba_start`:**

| tier | seed 42 | seed 43 | seed 44 | mean [min-max] |
|---|---:|---:|---:|---|
| T0 Viridian | 5/5 | 1/5 | 5/5 | 0.73 [0.20-1.00] |
| T2 Viridian | 5/5 | 0/5 | 3/5 | 0.53 [0.00-1.00] |
| T0 Viridian Forest | 5/5 | 0/5 | 5/5 | 0.67 [0.00-1.00] |
| T2 Viridian Forest | 5/5 | 0/5 | 0/5 | 0.33 [0.00-1.00] |
| T0 Pewter | 2/5 | 0/5 | 0/5 | 0.13 [0.00-0.40] |
| T2 Pewter | 0/5 | 0/5 | 0/5 | 0.00 [0.00-0.00] |

Pooled over all 15 episodes, with a two-sided Fisher exact test:

| milestone | T0 | T2 | p |
|---|---:|---:|---:|
| Route 1 | 15/15 | 14/15 | 1.00 |
| Viridian City | 11/15 | 8/15 | 0.45 |
| Viridian Forest | 10/15 | 5/15 | 0.14 |
| Pewter City | 2/15 | 0/15 | 0.48 |
| Route 22 | 1/15 | 5/15 | 0.17 |

**There is no detectable difference between pixels-only and all-RAM
observation.** T0's mean is higher on the main path (Viridian, Forest,
Pewter); T2's is higher on Route 22. Neither ordering clears the noise, and
the within-tier spread is the dominant effect: T0's Viridian rate ranges from
0.20 to 1.00 across seeds, T2's from 0.00 to 1.00. The closest call is
Viridian Forest, 10/15 vs 5/15, and p=0.14 does not support a claim.

This is a null result, and a useful one: minimal observation costs nothing
measurable at 20M steps. It is *not* evidence that RAM observations are
useless, for two reasons. T2's 2560-bit event vector may simply be unusable by
an MLP of this size (CJBoey reports needing a modular network for comparable
inputs), and a single 20M-step budget at 13% of the reference project's
Gym-2 compute may be too early for the difference to show.

T1 remains n=1: entropy collapsed to 0.42 nats and it never left Route 1. That
looks like a policy pathology and is recorded, not explained.

**Zero badges in all 30 episodes.** Pewter City is not Brock. The reference
project needed ~150M steps for Gym 2, so 20M is roughly 13% of that.

## Next

Replicate T0 and T2 with training seeds 43 and 44 (~10 h):

```powershell
.venv\Scripts\python scripts/run_tiers.py "C:/Users/neoga/roms/Pokemon Red.gb" `
    --tiers t0_pure t2_pragmatic --seeds 43 44 `
    --steps 20000000 --save-state "C:\Users\neoga\roms\states\bulba_start.state"
```

`scripts/run_tiers.py` runs the matrix sequentially, skips sessions that
already reached the target, resumes partial ones, and prints a per-run summary.
Add `--dry-run` to see the plan first. T1 is deliberately excluded: its
collapse looks like a policy pathology, so replicating it would mostly confirm
that one broken configuration is reproducibly broken.

Then evaluate each run at 5 seeds with `--save-state` and compare per tier
across seeds, reporting the spread rather than a single ordering.

Open questions:

- Is T1's collapse systematic? It may need a higher entropy coefficient, which
  would make it a hyperparameter comparison rather than an observation one.
- Is T2's 2,560-bit event vector usable by this MLP at all? CJBoey reports
  needing a modular network for comparable inputs.
