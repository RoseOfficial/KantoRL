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

## Running a long experiment

Always go through `scripts/run_tiers.py`. It starts the priority governor, so a
sweep yields CPU while you are using the machine and returns to full speed when
you are not. A run started any other way competes with whatever you are doing,
and a run throttled by hand stays throttled — Windows priority classes are
static, and nothing restores them.

```powershell
.venv\Scripts\python scripts/run_tiers.py "C:/Users/neoga/roms/Pokemon Red.gb" `
    --tiers t0_pure t2_pragmatic --seeds 42 43 --steps 20000000 --tag _mod `
    --save-state "C:/Users/neoga/roms/states/bulba_start.state"
```

- `--tag` names the sessions, so a new variant does not collide with old runs.
- `--dry-run` prints the plan without starting anything.
- The governor is on by default (6 h per queued run, 120 s idle threshold).
  `--no-governor` disables it.
- On exit the governor restores normal priority, so a sweep can never leave the
  machine throttled.

Evaluate with the same save state, or the policy is scored on a starting state
it never trained on:

```powershell
.venv\Scripts\python scripts/eval_runs.py "C:/Users/neoga/roms/Pokemon Red.gb" `
    --runs "runs/*_20M_mod_s*" `
    --save-state "C:/Users/neoga/roms/states/bulba_start.state"
```

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

## Matched comparison: the modular encoder changes the answer

Both tiers retrained with the same architecture, reward (`heal_weight=0`), start
state and budget. Two training seeds per tier, five evaluation episodes each.

| milestone | T0 pixels | T2 all RAM | Fisher p |
|---|---:|---:|---:|
| Route 1 | 5/10 | 10/10 | 0.033 |
| Viridian City | **0/10** | **10/10** | 1.1e-05 |
| Route 2 | 0/10 | 9/10 | 1.2e-04 |
| Viridian Forest | 0/10 | 8/10 | 7.1e-04 |
| Route 22 | 0/10 | 8/10 | 7.1e-04 |
| badges | 0 | 0 | - |

The stock-encoder comparison found no difference at all. With an encoder that
can use its inputs, the RAM-rich tier reaches Viridian City in every episode
while the pixel-only tier reaches it in none.

Two things are worth separating:

1. **The mean moved.** T2 reaches deeper milestones far more often.
2. **The variance collapsed.** T2's two seeds are nearly identical
   (5/5 and 5/5 to Viridian; 4/5 and 4/5 to the forest). T0's seeds disagree
   completely (5/5 vs 0/5 on Route 1). Richer observation did not just raise
   average performance, it made behaviour reproducible.

That second point matters for the research question: an observation prior is
worth more than its effect on the mean, because it also removes the run-to-run
lottery that made earlier single-seed results meaningless.

Caveats: two seeds per tier (n=10 episodes), no badges, and still only 13% of
the compute the reference project needed for Gym 2. The effect on Viridian is
large enough that more seeds would mainly tighten the estimate; the badge-level
question remains open.

## Which observation carries the effect?

Four matched arms, same reward (`heal_weight=0`), start state, horizon and
budget (20M). T0/T2 use two training seeds; the ablation arms one seed each.

| milestone | pixels | screens+map | all-but-map | all seven |
|---|---:|---:|---:|---:|
| Viridian City | 0/10 | **5/5** | 2/5 | **10/10** |
| Route 2 | 0/10 | 3/5 | 0/5 | 9/10 |
| Viridian Forest | 0/10 | 2/5 | 0/5 | 8/10 |
| Route 22 | 0/10 | 3/5 | 2/5 | 8/10 |
| badges | 0 | 0 | 0 | 0 |

Significance against the full tier on Viridian City:

- pixels 0/10, p < 1e-4
- screens+map 5/5, p = 1.00 (indistinguishable)
- all-but-map 2/5, p = 0.022

Reading:

1. **The map is the single biggest contributor.** Adding it to pixels takes
   Viridian from 0/10 to 5/5, matching the full tier. Nothing else tested does
   that.
2. **The map alone does not reproduce the full tier's depth.** Viridian Forest
   2/5 versus 8/10, Route 2 3/5 versus 9/10. Neither gap is individually
   significant at n=5, but the direction is consistent across both.
3. **The RAM fields cannot substitute for the map.** Without it they reach
   Viridian 2/5 but never Route 2 or the forest (0/5 each). Spatial memory
   appears to be what makes navigation past the first route work at all.

So the two groups are complementary rather than hierarchical: spatial memory
enables reaching the second town, and the event/level/badge observations add
depth beyond it. Neither subset alone matches the full tier.

Caveat: the ablation arms are one training seed each (n=5 episodes), so the
differences in depth are suggestive, not established. Replicating the two arms
is the obvious next step if that distinction matters.

Still true everywhere: **zero badges**.

## The 100M gate: an optimizer failure, not a capability ceiling

One seed, 100M steps, `t2_pragmatic`, `bulba_start`, 16.3 hours. Result: zero
badges, and the final policy could not reach Viridian City in 5/5 evaluation
episodes. The 20M baseline of the same tier reaches Viridian 5/5 and the forest
4/5.

The run did not spend 100M steps failing to learn; it learned, broke, and then
sat still:

| step | approx_kl | clip fraction | entropy | unique coords |
|---:|---:|---:|---:|---:|
| 8,192 | 0.0105 | 0.031 | 2.07 | 15 |
| 10,000,000 | **0.064** | **0.244** | 0.95 | 585 |
| 25,000,000 | 5.2e-4 | 0.001 | 1.96 | 153 |
| 50,000,000 | 8.8e-7 | 0.000 | 1.84 | 221 |
| 100,000,000 | 1.6e-6 | 0.000 | 1.85 | 215 |

`clip_range` bounds each sample's ratio, not the size of the whole update. At
~10M the run took an update that peaked at 0.093 KL (8x the 0.011 measured on
the 20M baseline at the same step). The policy first sharpened to entropy 0.95,
then lost the behaviour entirely: by 25M entropy was 1.96 of a 2.08 maximum and
KL had collapsed to 5e-4. A near-uniform policy cannot chain the ~50 actions
needed to leave Pallet Town; with no directed behaviour it earns no reward, so
the gradient stays near zero and the policy never sharpens again. 75M steps
(~12 hours) produced nothing.

Evaluating the run's own 10M checkpoint confirms the regression: Viridian 1/5,
Route 2 1/5 at 10M, versus 0/5 for both at 100M. The final model farms Pallet
Town house interactions (Town Map, Daisy) at steps 12k-16k.

Three fixes, all harness changes rather than new game priors:

1. **`target_kl: 0.02`** in every tier. SB3 stops an epoch once approximate KL
   exceeds 1.5x this value. Off by default in `KantoConfig` so existing
   checkpoints and configs keep their behaviour.
2. **`first_seen.csv`**, written during training by `FirstSeenCallback`: the
   step at which each badge and event flag first fires. Final-only evaluation
   cannot distinguish a run that regressed from one that never progressed.
   The nearest preceding checkpoint (every 100k) is the model that first
   reached the milestone.
3. **`interact_once_per_run: true`** in every tier. Interaction novelty used to
   reset each episode, so Pallet Town's NPCs paid again every life -- the same
   farmable pattern as the healing reward. It now pays once per position per
   environment process.

Caveat: this is one seed. Two runs with the same seed are not bit-identical
(their 100k checkpoints differ), so the comparison is one draw against another,
not a controlled replay. The internal signature -- a single oversized update
followed by 75M steps of near-zero KL -- is what makes the diagnosis, not the
seed count.

Next: three seeds at 20M with the guard, scored on forest, Pewter and the three
forest trainers. That is the current frontier: the 20M baseline enters the
forest 4/5 and beats all three trainers in 1/5.

## 20M with the guard: stable, but still short of the gym

Two seeds finished 20M with `target_kl: 0.02` and run-scoped interaction
novelty. Both produce the same phenotype:

| milestone (eval, 5 episodes) | baseline `_mod_s42` | guard s42 | guard s43 |
|---|---:|---:|---:|
| Viridian City | 5/5 | 5/5 | 5/5 |
| Route 2 | 4/5 | 5/5 | 5/5 |
| Viridian Forest | 4/5 | 5/5 | 5/5 |
| Pewter City | 0/5 | 1/5 | 1/5 |
| Pewter Gym | 0/5 | 0/5 | 0/5 |
| badges | 0 | 0 | 0 |

Training-time `first_seen.csv` shows the agents got further than eval
reproduces:

| | s42 | s43 |
|---|---:|---:|
| all three forest trainers | 4.45M | 6.07M |
| Bought Museum Ticket | 12.46M | 6.43M |
| Route 22 rival (1st) | 12.45M | 15.60M |
| Beat Pewter Gym Trainer 0 | 17.60M | 18.31M |

Neither final policy reaches the gym in evaluation. Seed 42's one Pewter
episode went to the museum, house and Pokémon Center; seed 43's went to Pewter
House 2 and the Pokémon Center. Both are sightseeing, not the gym.

Reading: `first_seen.csv` is a discovery log, not a capability measurement. A
gym-trainer event in one of 32 training environments is real, but it does not
mean the surrounding checkpoint can reproduce it. The final 20M policy reliably
navigates to the forest and only rarely pushes into Pewter.

The 20M phenotype is now consistent across seeds, which is itself a change: the
pre-guard runs varied much more. The remaining gap is the gym and the leveling
behind it, not exploration. The next step is to resume all three runs to 40M
and see whether Pewter reliability and the gym follow.

## First badge: seed 44 at 14.9M -- earned once, not retained

`first_seen.csv` for seed 44:

| milestone | step |
|---|---:|
| Beat Viridian Forest Trainer 0 | 2,194,080 |
| Beat Viridian Forest Trainer 1 | 2,484,864 |
| Beat Viridian Forest Trainer 2 | 6,159,744 |
| Bought Museum Ticket | 6,830,688 |
| **Beat Brock** | **14,919,776** |
| **Boulder Badge** | **14,921,952** |
| Got TM34 | 14,921,952 |

That is the first badge any configuration has earned, and it took 14.9M steps.
But the final 20M checkpoint cannot reproduce it: Viridian 5/5, Route 2 2/5,
Viridian Forest 2/5, Pewter 0/5, badges 0/5. It is the weakest of the three
final policies, not the strongest.

So the result is a **discovery, not a retained capability**. The policy found
Brock mid-run, then drifted away from the behaviour before the budget ended.
This is the same oscillation that made the 17.6M and 18.3M gym-trainer events
unreproducible for seeds 42 and 43, now with a badge attached.

One harness bug surfaced alongside it: `game/badges_max` was computed as the
maximum of a buffer that is cleared at every log, so the metric reported 0 for
the entire run even after the badge fired. It is now an all-time maximum, and
`first_seen.csv` is the authoritative record either way.

The three seeds now have a clean 20M midpoint:

| eval (5 episodes) | s42 | s43 | s44 |
|---|---:|---:|---:|
| Viridian City | 5/5 | 5/5 | 5/5 |
| Route 2 | 5/5 | 5/5 | 2/5 |
| Viridian Forest | 5/5 | 5/5 | 2/5 |
| Pewter City | 1/5 | 1/5 | 0/5 |
| badges | 0 | 0 | 0 |
| first badge during training | -- | -- | **14.9M** |

Next: resume all three from 20M to 40M, unchanged, and evaluate both the final
checkpoint and the first-badge checkpoint. If seed 44 stays regressed at 40M,
more budget is not the fix; checkpoint retention or reward shape is.

## 40M: navigation improves, the gym fight does not

Final 40M evaluation (5 episodes each):

| milestone | s42 | s43 | s44 |
|---|---:|---:|---:|
| Viridian City | 5/5 | 5/5 | 5/5 |
| Route 2 | 5/5 | 5/5 | 2/5 |
| Viridian Forest | 5/5 | 5/5 | 2/5 |
| Pewter City | 1/5 | 1/5 | 0/5 |
| badges | 0 | 0 | 0 |

That is the same phenotype as 20M. But the finals hide a large oscillation:

- seed 44 at **15M**: Pewter City 4/5, Pewter Gym 2/5
- seed 44 at **37M**: Pewter City 5/5, Pewter Gym 5/5
- seed 44 at **40M final**: Viridian Forest 2/5, Pewter City 0/5

So the run contained a much better navigator at 37M than at 40M. More budget did
not retain it.

Larger evaluations of the two interesting checkpoints:

| checkpoint | episodes | Pewter City | Pewter Gym | gym trainer beaten | badges |
|---|---:|---:|---:|---:|---:|
| 14.9M | 10 x 40k | 10/10 | 10/10 | 1/10 | 0/10 |
| 37M | 20 x 20k | 18/20 | 18/20 | 1/20 | 0/20 |

The agents reliably reach the gym and almost never win the fight. This is not
navigation and not the evaluation budget.

Party levels at episode end on the 37M checkpoint (5 episodes):

| seed | steps | party | level_sum | max_level |
|---:|---:|---:|---:|---:|
| 0 | 20,000 | 2 | 8 | 8 |
| 1 | 20,000 | 1 | 10 | 10 |
| 2 | 20,000 | 3 | 28 | 22 |
| 3 | 20,000 | 2 | 13 | 13 |
| 4 | 20,000 | 2 | 8 | 8 |

Brock's Onix is level 14. Most lives arrive at 8-13; one grinds to 22. The
training badge at 14.9M was almost certainly that rare high-level run, which is
why no checkpoint reproduces it reliably.

Conclusion: the binding constraint is **leveling**, not exploration. The level
reward exists but pays only `reward_scale * level_weight = 0.5` per level below
15, against 4.0 per new event. The next experiment raises `level_weight` and
measures whether the level distribution and gym win rate move.

The level reward formula and the cap are unchanged; only its weight changes.

## Level weight 4.0: negative result

Three seeds, 20M each, identical to the guard tier except `level_weight: 4.0`.

| final eval (5 episodes) | s42 | s43 | s44 |
|---|---:|---:|---:|
| Viridian City | 4/5 | 0/5 | 0/5 |
| Route 2 | 4/5 | 0/5 | 0/5 |
| Viridian Forest | 2/5 | 0/5 | 0/5 |
| Pewter City | 0/5 | 0/5 | 0/5 |
| badges | 0 | 0 | 0 |

Seed 42's levels were bimodal: episode maxima of 7, 37, 3, 22, 9. Seeds 43
and 44 did not merely fail to level; they stopped playing. Every seed-44
episode truncated at exactly 10,241 steps on 2 exploration tiles at level 6,
and seed 43 ended at 10-20k steps on 2-9 tiles. Those are dead policies, not
underlevelled ones.

Why: the level potential is the **sum of party levels**, so catching a new
Pokemon jumps it. At weight 4.0 a single catch can pay roughly 10 reward,
against 4.0 for a new event. The logs show the corresponding instability:
KL spikes to 0.08, value loss to 0.22, and the guard firing repeatedly. Two
seeds collapsed; the third over-grinded and lost navigation.

So the experiment did not test "pay more for leveling" cleanly: raising the
weight also raised the catch bonus, which was never the intended signal.

Conclusion: **do not use level_weight 4.0.** Revert to 1.0 for capability runs.
If leveling is pursued again, the potential should key on the strongest party
member's level (or the starter's), not the sum, so catching a Pokemon does not
pay a windfall.

## Max-level potential: first reliably reproducible badge

`t2_levelmax` -- strongest party member, continuous curve, weight 4.0 -- at 20M,
three seeds.

20k evaluation:

| milestone | s42 | s43 | s44 |
|---|---:|---:|---:|
| Viridian City | 5/5 | 5/5 | 5/5 |
| Route 2 | 5/5 | 5/5 | 5/5 |
| Viridian Forest | 5/5 | 5/5 | 5/5 |
| Pewter City | 5/5 | 5/5 | 0/5 |
| Pewter Gym | 4/5 | 5/5 | 0/5 |
| Boulder Badge | 0/5 | **1/5** | 0/5 |

Training `first_seen.csv`:

| | s42 | s43 | s44 |
|---|---:|---:|---:|
| Beat Pewter Gym Trainer 0 | 14.77M | 10.14M | -- |
| Beat Brock | 16.67M | 10.48M | -- |
| Boulder Badge | 16.67M | 10.48M | -- |

Two of three seeds earned a badge during training, and seed 43's final policy
reproduces it. That is the first final-checkpoint badge in the project.

### The 40k evaluation changes the number

Ten episodes per seed, 40k steps each:

| milestone | s42 | s43 |
|---|---:|---:|
| Pewter City | 10/10 | 10/10 |
| Pewter Gym | 10/10 | 9/10 |
| gym trainer beaten | 1/10 | **9/10** |
| Beat Brock | 0/10 | **7/10** |
| median badge step | -- | 23,038 |

Seed 43's policy is a solo starter at level 13-15 and beats Brock in 70% of
lives. Seed 42's lives often carry 2-4 Pokemon and reach level 26-34, but still
lose the gym trainer; catching and overlevelling did not translate into the
right battle.

### Why the 20k numbers were low

Training episodes truncate at `max_steps: 20480`. Seed 43's median badge step
is 23,038 -- beyond the training horizon. The training loop never sees most of
the successful trajectories, and the 20k evaluation was measuring a shorter
game than the policy can play.

So the max-level reward worked: no dead policies, a clean level signal, and a
reproducible badge. The next constraint is the horizon, not the reward. Raise
`max_steps` to 40,960 and `stagnation_steps` to 20,480 so training can see the
badge and continue past it.

## 40k horizon: negative result on seed 42

Seed 42 finished 20M with `max_steps: 40960`. Final evaluation, 10 episodes
of 40k:

| metric | result |
|---|---:|
| Pewter City | 0/10 |
| Pewter Gym | 0/10 |
| Boulder Badge | 0/10 |
| events per episode | 15 |
| exploration tiles | 18-28 |
| party levels (max) | 17-44 |

The policy grinds in the early routes and never leaves them. Compare the same
seed at the 20k horizon: Pewter City 5/5, Pewter Gym 4/5. Longer episodes made
it substantially worse.

Why: a 40k episode gives the agent twice as long to farm the level reward
before the episode ends. The above-15 potential still pays 0.5 per level at
weight 4.0, and the agent can collect that in Route 1 indefinitely. The horizon
change amplified the grind instead of letting the policy reach Brock.

Conclusion: **do not lengthen episodes for this reward.** Keep `max_steps:
20480`. The best configuration remains the 20k-horizon max-level tier, where
seed 43's final policy beats Brock 7/10 at a 40k evaluation. The horizon
mismatch is real, but fixing it by lengthening training episodes trades a
measurement artifact for a much worse policy.

## 40M extension: seed 43 regressed from its 20M peak

The 20M final of seed 43 is the best policy in the project: Brock 7/10 at a 40k
evaluation. Extending that same run to 40M destroyed it.

| checkpoint | Pewter City | Pewter Gym | Brock | badge |
|---|---:|---:|---:|---:|
| 20M | 10/10 | 9/10 | **7/10** | 7/10 |
| 40M | 10/10 | 6/10 | **0/10** | 0/10 |

Training did pass Brock between the two: `first_seen.csv` records Route 3
trainers beaten at 25.8M and 30.4M. The capability existed mid-run and was gone
by the endpoint.

So more training made the best policy worse. The bottleneck is **checkpoint
retention**, not budget. The 20M checkpoint remains at

    runs/t2_levelmax_40M_lvlmax_s43/checkpoints/model_20000768.zip

(the session was renamed when it was extended). That is the deliverable: a
final policy that beats Brock in 70% of evaluated lives at a 40k horizon.

Do not extend it further. Future work should either preserve the best
checkpoint automatically or stabilize the policy so the capability survives to
the endpoint; adding steps without either just loses it.

## Best-checkpoint retention

`FirstSeenCallback` now snapshots the policy when it reaches a new milestone:

- `checkpoints/best_model.zip` -- overwritten whenever a new event or badge is
  first seen, with `best_model_step.txt` naming the step and milestone.
- `checkpoints/badge_<n>_model.zip` -- saved at the first n-th badge and **never
  overwritten**, so the policy that first beat a gym leader survives later
  regressions.
- Matching `_vecnormalize.pkl` files for resume. Evaluation does not need them:
  `norm_obs=False`, so only reward normalization is included.

Evaluate the best snapshot with:

    python scripts/eval_runs.py "rom.gb" --runs "runs/<tier>_s*" --best ...

which writes `eval_best/` instead of overwriting `eval/`.

Why this exists: the 20M seed-43 policy beats Brock 7/10, while the same run at
40M beats him 0/10. Nothing preserved the 20M peak, so extending the run
destroyed the best model the project had. Retention is now automatic; adding
steps no longer risks losing the policy that earned the badge.

## Checkpoint frontier: 31M reaches Route 3

Evaluating the 40M extension's mid-run checkpoints at 40k x 10 episodes:

| checkpoint | Pewter | Pewter Gym | gym trainer | Brock | Route 3 | Route 3 trainers |
|---|---:|---:|---:|---:|---:|---:|
| 20M | 10/10 | 9/10 | 9/10 | **7/10** | 0/10 | 0/10 |
| 26M | 10/10 | 10/10 | **10/10** | 4/10 | 2/10 | 0/10 |
| **31M** | 6/10 | 6/10 | 4/10 | 3/10 | **3/10** | **3/10 each** |
| 35M | 10/10 | 9/10 | 1/10 | 0/10 | 0/10 | 0/10 |

No single checkpoint dominates. The 20M policy is the badge-1 specialist; the
26M policy clears the gym trainer most reliably; the 31M policy is the first to
reach Route 3 and beat all three Route 3 trainers in evaluation.

This is the oscillation in checkpoint form. The 40M run did contain a better
post-Brock policy than the 20M run -- it just did not survive to the endpoint.
`best_model.zip` now captures exactly this kind of mid-run peak when the
Route 3 events first fire.

For "furthest progress", 31M is the new frontier: 3/10 Brock and 3/10 Route 3,
with all three Route 3 trainers beaten in those lives. For "fewest steps to
badge 1", 20M remains the answer at 7/10.

## Lower learning rate: 100% Brock and 100% Route 3

Resumed seed 43 from the 31M Route 3 checkpoint at 1e-4 for 9M steps
(`t2_levelmax_lr1e4`). Final 40M evaluation, 10 episodes of 40k:

| milestone | result | median step |
|---|---:|---:|
| Boulder Badge | **10/10 (100%)** | 6,748 |
| Beat Brock | 10/10 | 6,717 |
| Pewter Gym trainer | 10/10 | 6,748 |
| Pewter City | 10/10 | 3,121 |
| Route 3 | **10/10 (100%)** | 7,852 |
| Route 3 trainer 0 | 7/10 | 15,866 |
| Route 3 trainer 1 | 5/10 | 33,469 |
| Route 3 trainer 2 | 6/10 | 35,899 |
| Route 3 trainer 6 | 0/10 in eval | first seen at 38.2M |

Training `badges_mean` rose to 0.75 by the end of the run, and the learning
rate stayed at 1e-4 throughout. The lower rate did not merely retain the Route
3 policy: it made the badge reliable and cut the median badge step from 23,038
at 31M to 6,748 here.

The retained `best_model.zip` (saved when Route 3 trainer 6 was first beaten at
38.2M) is also 10/10 on the badge, but slower (median 12,454) and weaker on
Route 3 trainers 1 and 2 (30% and 10% versus 50% and 60%). The final 40M
checkpoint is the better policy; retention did its job by preserving the
alternative, not by replacing the final.

This is the first reliable badge and the first reliable Route 3 in the project.
The remaining wall is the last stretch of Route 3 and Mt. Moon: trainer 6 has
been beaten in training but not in any evaluated episode.

## 50M: the Route 3 trainer wall moves again

50M final evaluation, 10 episodes of 40k:

| milestone | result | median step |
|---|---:|---:|
| Boulder Badge | **10/10 (100%)** | 6,574 |
| Beat Brock | 10/10 | 6,543 |
| Route 3 | **10/10 (100%)** | 7,722 |
| Route 3 trainer 0 | 9/10 | 11,322 |
| Route 3 trainer 1 | 9/10 | 14,217 |
| Route 3 trainer 2 | 8/10 | 17,626 |
| Route 3 trainer 3 | 8/10 | 21,378 |
| Route 3 trainer 4 | 7/10 | 23,302 |
| Route 3 trainer 5 | 4/10 | 24,711 |
| Route 3 trainer 6 | 7/10 | 23,609 |
| Route 3 trainer 7 | 4/10 | 22,691 |

Mt. Moon and Cerulean are still 0/10. The last two Route 3 trainers are the
gate.

Training `badges_mean` fell to 0.19 in the final log window, which looked like
a regression. It was not: the final 50M policy is better than 40M on every
Route 3 trainer and slightly faster to the badge. The windowed metric is
misleading; the evaluation is the measurement.

The retained `best_model` (Route 3 Trainer 7 at 49.91M) is also 10/10 on the
badge but slower and weaker on trainers 2/5; the final checkpoint is the better
artifact.

## 60M: the cap binds, and retention saves the peak

60M final evaluation, 10 episodes of 40k:

| milestone | 50M final | 60M final | retained best (49.91M) |
|---|---:|---:|---:|
| Boulder Badge | 10/10, 6,574 | 9/10, 18,796 | **10/10, 7,333** |
| Route 3 | 10/10 | 9/10 | 10/10 |
| Route 3 trainer 3 | 80% | 80% | 80% |
| Route 3 trainer 5 | 40% | **80%** | 10% |
| Route 3 trainer 6 | 0% | 0% | **50%** |
| Route 3 trainer 7 | 0% | 0% | **50%** |

The 60M final regressed on badge speed and lost trainers 6/7; the retention
snapshot is the better artifact. Trainers 6 and 7 have median first steps of
28,917 and 29,210 in the best model -- above the 20,480 training cap. Training
only sees the rare attempts that reach them before truncation.

This is the trigger agreed for the horizon change: raise `max_steps` to
40,960, keep the 1e-4 learning rate, and seed from the retained best rather
than the regressed final. Mt. Moon and Cerulean remain 0/10.

## 40k horizon: Route 3 clears, Mt. Moon opens

Seeded the retained 49.91M best into `t2_levelmax_h40_lr1e4` (40,960 cap,
20,480 stagnation, 1e-4) and trained 10M steps to 60M.

Final 60M, 10 episodes of 40,960:

| milestone | result | median |
|---|---:|---:|
| Boulder Badge | 9/10 | 13,593 |
| Route 3 | 9/10 | 13,876 |
| Route 3 trainer 6 | 7/10 | 18,494 |
| Route 3 trainer 7 | 7/10 | 20,193 |
| Mt. Moon 1 trainer 3 | 7/10 | 27,814 |
| Mt. Moon 1 trainer 4 | 7/10 | 26,406 |

Retained best (59.83M, first Mt. Moon 3 Trainer 1):

| milestone | result | median |
|---|---:|---:|
| Boulder Badge | 9/10 | 11,680 |
| Route 3 trainer 6 | **9/10** | 15,967 |
| Route 3 trainer 7 | **9/10** | 16,821 |
| Mt. Moon 1 trainer 4 | **9/10** | 18,652 |
| Mt. Moon 1 trainer 5 | **9/10** | 18,952 |
| Mt. Moon 3 trainer 1 | 1/10 | 26,755 |

Cerulean and Misty remain 0/10.

The 20k cap was the binding constraint: doubling it took Route 3 trainers 6
and 7 from 0-50% to 70-90% and carried the policy into Mt. Moon, where it now
beats late trainers. The badge rate dipped from 10/10 to 9/10 and the median
slowed, a deliberate trade of early-game speed for post-Brock depth.

The next step is the same pattern: seed a 70M continuation from the 59.83M
best and target the Mt. Moon exit and Cerulean.
