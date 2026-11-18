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

## Are the extra observations actually used?

A null tier result only means something if the richer tier's policy uses the
extra inputs. `scripts/probe_obs_sensitivity.py` collects real observations,
randomises one component at a time, and measures how far the action
distribution moves.

250 transitions, T2 checkpoint (`model_20000768`), seed 42 and seed 44:

| component | mean KL (s42 / s44) | top-1 action changed |
|---|---|---|
| map (48x48) | 18.9 / 12.0 | 96% / 36% |
| screens | 1.63 / 0.22 | 78% / 50% |
| events (2560 bits) | 0.0001 / 0.0002 | **0% / 0%** |
| badges (8) | 0.0000 / 0.0000 | **0% / 0%** |
| level (8) | 0.0000 / 0.0000 | **0% / 0%** |
| health (1) | 0.0000 / 0.0000 | **0% / 0%** |
| recent_actions (3) | 0.0001 / 0.0007 | **0% / 0%** |

The control is T0, whose only input is screens: randomising it moves 93% of
actions. So the probe detects genuine dependence.

**Five of T2's seven inputs are ignored entirely.** T2 is effectively
"pixels plus a map". The tier comparison therefore measured the encoder, not
the value of RAM observations. It also explains T1: health is ignored, so its
input is functionally identical to T0's, and its collapse was training
variance rather than an observation effect.

This matches CJBoey's report that a modular network is required for these
inputs ("without this network architecture, the agent fails to learn anything
when given the same observation").

`kantorl.encoders.ModularExtractor` addresses it: one encoder per component
(NatureCNN for screens, Linear/LayerNorm/Linear for each vector), fused to a
fixed width so every tier presents the same capacity to the policy. It is now
the default for new models; `create_policy_kwargs(modular=False)` restores the
stock extractor. Checkpoints trained before this change continue to load with
their own extractor.

Not yet demonstrated: that training with the modular encoder actually makes
the policy use the auxiliary inputs. That needs a run.

## T1's collapse

T1's single seed reached a final entropy of 0.42 nats (max 2.08) and never left
Route 1. Given that health is one of the ignored inputs above, T1's input is
functionally the same as T0's, so this is most likely a training pathology
rather than an observation effect. Untested.

## Reward exploit: healing

The first 20M run with the modular encoder earned ~490 reward per episode while
reaching only Route 1, versus ~25 for the stock-encoder runs. A component
breakdown over 3,000 steps shows why:

| component | reward |
|---|---:|
| heal | 65.0 |
| event | 12.0 |
| map_progress | 4.0 |
| explore | 1.1 |
| interact | 0.8 |

Healing was 78% of the total. The exploit is: lose a wild battle, black out,
have the party restored, collect +5, repeat. The agent spent the episode near
the player's house doing exactly that.

The stock encoder never found this because it could barely act. Better
perception made the agent capable enough to discover the exploit, which is
worth remembering: reward farms surface as policies improve.

`heal_weight` now defaults to 0. Healing is our own shaping term, not a
frontier prior (neither PokeRL nor CJBoey reward it), so removing it costs
nothing that the task requires. It remains opt-in for anyone who has accounted
for the exploit.

Any run trained before this fix optimised the exploit and is not comparable.
That includes `t2_pragmatic_mod_s42`.
