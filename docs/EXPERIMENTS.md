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

## 70M: two trainers into Mt. Moon 3, badge rate restored

Final 70M, 10 episodes of 40,960:

| milestone | result | median |
|---|---:|---:|
| Boulder Badge | **10/10** | 12,470 |
| Route 3 | 10/10 | 12,906 |
| Route 3 trainer 6 | 9/10 | 15,790 |
| Route 3 trainer 7 | 9/10 | 16,476 |
| Mt. Moon 1 trainer 4 | 9/10 | 18,677 |
| Mt. Moon 1 trainer 5 | 9/10 | 22,667 |
| Mt. Moon 3 trainer 1 | 6/10 | 25,936 |
| Mt. Moon 3 trainer 2 | 0/10 in eval | first seen 66.25M |

New `first_seen` after the 59.83M seed:

- Mt. Moon 1 Trainer 1 at 64.89M
- Mt. Moon 3 Trainer 2 at 66.25M

The retained best at 66.25M is stronger on Route 3 (trainer 6 at 100%, trainer
7 at 90%) but weaker on Mt. Moon 3 (10% versus the final's 60%), so the final is
the better frontier base. Badge rate is back to 10/10 after the 40k horizon
dip. The Exit Super Nerd and Cerulean remain unreached.

## 80M: regression at 40k, fall back to the 70M policy

80M final, 10 episodes of 40,960:

| milestone | 70M final | 80M final |
|---|---:|---:|
| Boulder Badge | 10/10, 12,470 | 10/10, 18,172 |
| Route 3 | 10/10 | 10/10 |
| Mt. Moon 1 trainer 4 | 90% | 10% |
| Mt. Moon 1 trainer 5 | 90% | 0% |
| Mt. Moon 3 trainer 1 | 60% | 10% |

The 80M wave kept the badge but lost almost all Mt. Moon depth. Its new event
was Mt. Moon 1 Trainer 0 at 75.12M, and the retained "best" at that step is only
3/10 on Brock -- a lucky event, not a better policy. The final is still 10/10,
but it is worse than the 70M final on every Mt. Moon metric.

Deepest medians at 80M are 34k+, close to the 40,960 cap. The next wave
re-seeds from the **70M final** and raises the cap to 51,200, keeping the 1e-4
rate. This is the first horizon change triggered by data rather than a fixed
schedule. The 80M session is kept as the regression record, not used as a base.

## 80M at 51k: the 78.8M checkpoint is the best policy; the final collapsed

The 51,200-horizon run was seeded from the 70M final and trained 10M steps.

| checkpoint | Boulder | Route 3 | Route 3 trainers 0/1/2/3/5/6 | Mt. Moon 1 t4 | Mt. Moon 3 t1 |
|---|---:|---:|---|---:|---:|
| 70M final | 10/10 | 10/10 | mixed, 6 at 90% | 90% | 60% |
| **78.8M** | **10/10** | **10/10** | **100% each** | **100%** | **50%** |
| 80M final | **0/10** | 0/10 | -- | -- | -- |

Training `badges_mean` fell from 0.56 at 78.8M to 0 by 79.8M: a late destructive
update, not a horizon failure. The 78.8M checkpoint is the strongest policy in
the project, with deep medians of 40.7k and 44.8k, close to the 51.2k cap.

No new event fired after 70M, so `best_model` never saved the 78.8M peak. This
is a retention gap: an event-based snapshot cannot see a policy that improves
without setting a new flag. The checkpoint survives only because cumulative
saves every 100k are kept.

Next: seed from 78.8M, lower the learning rate to 5e-5 to stop the late churn,
and raise the cap to 61,440 for the remaining Mt. Moon depth.

## 90M: 5e-5 stabilizes, Mt. Moon 3 Trainer 2 at 70%

The 61,440-horizon run at 5e-5 was stable: `badges_mean` stayed near 1.0,
with no late collapse. Final 90M, 10 episodes of 61,440:

| milestone | result | median |
|---|---:|---:|
| Boulder Badge | **10/10** | 6,844 |
| Route 3 | 10/10 | 7,226 |
| Route 3 trainers 0/1/2/3/4/6/7 | 10/10 each | 10.5k-21.6k |
| Route 3 trainer 5 | 5/10 | 30,172 |
| Mt. Moon 1 trainer 4 | 10/10 | 27,032 |
| Mt. Moon 3 trainer 1 | 5/10 | 38,085 |
| **Mt. Moon 3 trainer 2** | **7/10** | 45,174 |

The Exit Super Nerd and Cerulean remain 0/10. This is the strongest and fastest
policy so far: the badge median is back to 6,844, the full Route 3 trainer line
clears except trainer 5, and Mt. Moon 3 Trainer 2 is reached 7/10. The 5e-5 rate
is the right setting for this stage; the 1e-4 runs were churning.

## 100M: the 99M checkpoint is the best policy; the party reads were corrupt

The 100M wave resumed from the 90M final at 5e-5 with the 61,440 horizon and
added **no new first-seen events**. Training was unstable even at 5e-5:

- `badges_mean` fell from 1.0 to 0.06 at about 95.9M, recovered by 96.3M,
  then fell again at 99.83M and stayed there.
- Final 100M, 10 episodes of 61,440: Boulder **5/10** (median 28,164),
  Route 3 4/10, no Mt. Moon.
- A checkpoint sweep found the peak before the last collapse at **99.0M**.

The 99.0M checkpoint is the best policy in the project, evaluated under the
corrected party reads (below), 10 episodes of 61,440:

| milestone | result | median |
|---|---:|---:|
| Boulder Badge | **10/10** | 7,182 |
| Route 3 | 10/10 | 7,440 |
| Route 3 trainer 5 | 9/10 | 18,911 |
| Mt. Moon 1 trainer 2 | 10/10 | 21,928 |
| Mt. Moon 1 trainer 5 | 9/10 | 16,759 |
| Mt. Moon 1 trainer 6 | 8/10 | 30,826 |
| Mt. Moon 3 trainer 2 | **9/10** | 28,482 |
| Mt. Moon B2F | **10/10** | 24,587 |

The Exit Super Nerd and Cerulean remain 0/10. The 99.5M checkpoint still beat
Brock 10/10 in the same sweep but had already regressed in Mt. Moon (B1F 1/10,
no B2F), so depth was oscillating before the final collapse.

### The party reads were corrupt

A level-89 party member in an evaluation exposed a memory-layout bug:

- `get_party_levels` advanced **one byte per party slot**. Party members are
  44 bytes apart (`PARTY_MON_STRIDE`), so slots 2+ read max-HP, stat, and PP
  bytes as levels. This fed the `level` observation and the
  `level_source=max` reward; `max_level` could jump to a fake 89.
- `get_party_hp` advanced two bytes per slot and read 16-bit HP little-endian.
  Gen 1 stores HP high byte first, so 25 HP read as 6400. The early-game HP
  *ratio* was roughly preserved because both operands were byte-swapped, which
  is why the health observation looked plausible.

Fixed in `memory.py`: both readers use the 44-byte stride, `read_word` is
big-endian, and empty/phantom slots are skipped. `tests/test_memory.py` covers
the stride, byte order, fainted members, and empty slots. All 149 tests pass.

Every earlier checkpoint, including 99M, was trained with the corrupted level
signal. The 99M checkpoint survives the correction: under the fixed reads it
still scores Boulder 10/10 (median 7,182), Mt. Moon 3 Trainer 2 9/10, and B2F
10/10. The fix starts a cleaner baseline for the next wave; it does not
invalidate the best artifact.

### What this changes for the next wave

The last 1M steps of the 100M wave were destructive and added no event. The
next wave should resume from 99M with the fixed party reads at 5e-5, and stop
at the first sustained `badges_mean` collapse instead of running to the target.
If Mt. Moon still stalls, the remaining suspects are the coarse 9x9 global map
for indoor mazes and the level reward paying for battles rather than the
Mt. Moon exit.

## Development vs. final measurement

The 10M waves are a development process, not a single fixed-ruleset run. The
rules from here:

- Development uses seeds 0-9. Seeds 10-19 are held out for the final
  measurement and must not be used for checkpoint selection.
- Every change (horizon, learning rate, reward, observation, checkpoint
  selection) is logged with the evaluation that motivated it. Total compute
  counts all runs, not just the winning lineage.
- Checkpoint retention: the latest 100 checkpoints (~10M steps at 100k
  spacing) are kept for the current wave, every 1M checkpoint is kept for the
  lineage, and `best_model` / `badge_<n>_model` snapshots are permanent. At 1B
  steps this is ~35 GB instead of ~320 GB.
- The current 99M/104M lineage is development. The headline result is a clean
  fixed-ruleset run: fixed observation, reward, horizon and learning rate;
  pre-registered checkpoint selection; trained from scratch; evaluated on
  seeds 10-19.
- Trigger the clean control when the game is completed, or if the frontier
  stalls for three waves without a new event. In the stall case, use the
  control to test the suspect prior rather than adding more waves.
- Until then, any capability claim is "development lineage, N cumulative
  steps, human-staged."

The retrospective analysis (first-seen logs, evaluation reports, tier YAMLs,
TensorBoard) is observational: the waves change several things at once and
start from different checkpoints. It can identify which changes coincided with
progress, not prove which prior was load-bearing. The clean run is the causal
claim.

## 104M local map: one transient event, broad regression

The 5M fine-tune (99M -> 104M, 5e-5, 61,440) added a 15x15 visited-tile
window centered on the player. Training produced one new first-seen event:
`Beat Mt. Moon 3 Trainer 3` at 102,831,968, and `best_model` saved that
checkpoint.

Ten-seed evaluation, 61,440 steps, `bulba_start`:

| checkpoint | Boulder | Route 3 | Mt. Moon 3 t2 | Mt. Moon B2F | Trainer 3 |
|---|---:|---:|---:|---:|---:|
| 99M baseline | **10/10** | 10/10 | **9/10** | **10/10** | 0/10 |
| 104M final | 9/10 | 9/10 | 1/10 | 7/10 | 0/10 |
| 102.83M best | 9/10 | 9/10 | 0/10 | 0/10 | 0/10 |

The event was a training-only transient: the event-time `best_model` does not
reach Mt. Moon from a fresh start, and the final checkpoint regressed broadly.
The local-map fusion head grew 8x (norm 0.18 -> 1.61), but the first encoder
layer barely moved (21.21 -> 21.25), so the policy learned a small linear
readout of mostly-random local features while the underlying policy drifted.
At 5e-5 the fine-tune was destructive, the same pattern as the 80M and 100M
late collapses.

The 99M checkpoint remains the best artifact. The local map is not ruled out
as an observation, but this fine-tune is a negative result; the next probe
should change the reward balance, not add more map detail. The trace showed
the policy grinding wild battles on B2F while the event reward is 8.0 and the
per-tile exploration reward is 0.015, so lowering `level_weight` and raising
`explore_weight` is the cheaper next test.

## 104M reward rebalance: development probe

Started from the 99M best (legacy seven-component observation) with
`level_weight` 1.0, `explore_weight` 1.0 and a 2e-5 rate, same 61,440
horizon. Target: the B2F trace showed 212 wild battles and 8 levels gained
while the fossil room stayed unvisited; a level was worth 0.5 and a new tile
0.015. The rebalance makes the fossil room's ~16 new tiles worth about six
levels and halves the update rate to avoid the 5e-5 regression.

This is a development probe, not the clean fixed-ruleset run. Seeds 0-9 remain
the development set. Result pending.

## 104M reward rebalance: negative

The 5M probe (99M -> 104M, 2e-5, `level_weight` 1.0, `explore_weight` 1.0)
finished with `badges_mean` 1.0 but `events_mean` 24.3 and
`explore_tiles_mean` 590, below the 99M peak (~30 and ~1.4k). The only new
first-seen was an unnamed event flag (#1392) at 102,644,160; no curated
milestone.

Ten-seed evaluation, 61,440 steps:

| checkpoint | Boulder | Route 3 | Mt. Moon 3 t2 | Mt. Moon B2F |
|---|---:|---:|---:|---:|
| 99M baseline | **10/10** | 10/10 | **9/10** | **10/10** |
| 104M reward | 8/10 | 8/10 | 8/10 | 8/10 |

Lowering the level reward and raising the exploration reward did not buy
exploration; it degraded the core policy. The 99M checkpoint remains the best.
This is a negative result for the reward-rebalance axis.

The next mission-compatible probe is a method setting, not a new hand-authored
rule: raise gamma from 0.995 to 0.999 on the 99M policy. The current effective
horizon is ~200 steps, while the B2F loops span thousands; Boey uses
0.999/0.9996. If that fails, the next step is the minimal observation prior
(the player marker), not the full Boey minimap.

## 104M gamma probe: development

Started from 99M with gamma 0.999 and a 2e-5 rate. The effective horizon moves
from ~200 to ~1,000 steps; the B2F loops span ~1,000. Observation, reward and
horizon are unchanged. This is a method setting, not a new hand-authored rule;
the checkpoint's gamma is saved as 0.999 so PPO.load restores it. Seeds 0-9
remain the development set. Result pending.

## 104M gamma 0.999: positive method result, frontier unchanged

The 5M gamma probe (99M -> 104M, gamma 0.995 -> 0.999, 2e-5) finished with
`badges_mean` 1.0, `events_mean` 27.8 and `explore_tiles_mean` 914. The only
new first-seen was an unnamed event flag (#1400) at 99,902,176; no curated
milestone.

Ten-seed evaluation, 61,440 steps:

| checkpoint | Boulder | Route 3 | Mt. Moon 3 t2 | Mt. Moon B2F | Mt. Moon 1 t1/t5/t6 |
|---|---:|---:|---:|---:|---|
| 99M baseline | 10/10 (7,182) | 10/10 | 9/10 | 10/10 (24,587) | 10/10, 9/10, 8/10 |
| **104M gamma** | **10/10 (5,160)** | 10/10 | **10/10** | 10/10 (28,590) | 8/10, 7/10, 0/10 |
| 99.9M best | 9/10 | 8/10 | 5/10 | 8/10 | 5/10, 4/10, -- |

Gamma 0.999 is a positive method result: the badge and Route 3 are faster,
Mt. Moon 3 Trainer 2 becomes 10/10, and B2F stays 10/10. It loses Mt. Moon 1
depth (t6 0/10) and is slower at B2F, and it still does not reach Trainer 3,
the Exit Super Nerd, or Cerulean. The 99M checkpoint remains the deepest
policy; the gamma final is a strong speed/reliability alternative and a
candidate method setting for the final ruleset.

The next observation-prior test is the minimal player marker on the 99M base.
If we hold the line at zero new observation priors, the 99M is the ceiling and
the coarse map is the binding limit.

## 104M player marker: development

Started from 99M with `map_player_marker=True`: the player's current 48x48
cell is 128, against 0 = unvisited and 255 = visited. This is the minimal
localization prior: no walkability, sprites, warps, ledges or route
knowledge, and the observation shape is unchanged, so the 99M checkpoint
loads directly. Observation, reward, gamma and horizon match the 99M
baseline; the continuation rate is 2e-5. Seeds 0-9 remain the development
set. Result pending.

## 104M player marker: negative

The 5M marker probe (99M -> 104M, `map_player_marker=True`, 2e-5) finished
with `badges_mean` 1.0 and `events_mean` 27.2; no curated first-seen events.
Ten-seed evaluation with `--map-player-marker`:

| checkpoint | Boulder | Route 3 | Mt. Moon 3 t2 | Mt. Moon B2F | Mt. Moon 1 t1/t5/t6 |
|---|---:|---:|---:|---:|---|
| 99M baseline | 10/10 (7,182) | 10/10 | 9/10 | 10/10 | 10/10, 9/10, 8/10 |
| 104M gamma | 10/10 (5,160) | 10/10 | 10/10 | 10/10 | 8/10, 7/10, 0/10 |
| 104M marker | 10/10 (8,810) | 10/10 | 5/10 | 2/10 | 0/10, 1/10, 0/10 |

The marker kept the badge and improved Route 3 trainers 5/6, but collapsed
the Mt. Moon depth. It did not reach Trainer 3, the Exit Super Nerd, or
Cerulean. The minimal prior ladder is now exhausted: the local visited map,
the reward rebalance and the player marker all failed; gamma 0.999 is the
only positive method result. The coarse 48x48 global map is the binding limit.

The next step is either a deliberate larger observation prior (a local
tile-ID or walkability map, with no sprites/warps/ledges) or stopping the
prior ladder and running the clean fixed-ruleset measurement on the 99M or
gamma policy.

## Minimal-prior ceiling at 99M

The minimal-prior ladder stops here. The 99M policy remains the deepest:
Boulder 10/10 (median 7,182), Route 3 10/10, Mt. Moon 3 Trainer 2 9/10,
Mt. Moon B2F 10/10. Trainer 3, the Exit Super Nerd, and Cerulean remain
0/10.

The gamma-0.999 104M final is a speed/reliability alternative: Boulder 10/10
at 5,160, Route 3 10/10, Mt. Moon 3 Trainer 2 10/10, B2F 10/10, but it loses
Mt. Moon 1 depth (t1/t5/t6 8/10, 7/10, 0/10). The local visited map, the
reward rebalance, and the player marker all failed; gamma 0.999 is the only
positive change.

The binding limit is the coarse 48x48 global map: each cell is ~9x9 tiles,
and the fossil room is smaller than one cell. Minimal local maps and a player
marker do not fix the indoor maze. The next phase is either:

1. a clean fixed-ruleset run from scratch (held-out seeds 10-19) to confirm
   the development lineage, or
2. a deliberate tile-level map prior (tile IDs or walkability, no
   sprites/warps/ledges) as a new observation experiment.

Another small prior is not warranted. Artifacts: `runs/best/red_99M_s43.zip`
(deepest) and `runs/best/red_gamma_104M_s43.zip` (speed/reliability).

## 104M local tile map: development

Started from 99M with an opt-in 15x15 raw tile-ID window (`local_tiles`) read
from the visible 20x18 screen tilemap, one cell per tile, centered on the
player's sprite. The values are raw tileset tile IDs (walls, floor, doors,
water); no sprites, warps, ledges or route knowledge are added. The
observation shape grows by one 15x15 branch; the 99M policy is warm-started
through `scripts/transfer_local_map.py --new-component local_tiles` with
parity 0.0 on an empty window. Reward, gamma and horizon match the 99M
baseline; the rate is 2e-5. Result pending.

## 104M local tile map: negative under warm-start

The 5M tile probe (99M -> 104M, `local_tiles`, 2e-5) finished without a new
curated first-seen event and without a `best_model`. Ten-seed evaluation:

| checkpoint | Boulder | Route 3 | Mt. Moon 3 t2 | Mt. Moon B2F |
|---|---:|---:|---:|---:|
| 99M baseline | **10/10** (7,182) | 10/10 | **9/10** | **10/10** |
| 104M gamma | 10/10 (5,160) | 10/10 | **10/10** | 10/10 |
| 104M tiles | 6/10 (7,966) | 6/10 | 3/10 | 6/10 |

The warm-start parity on an empty window was 0.0, so the new branch starts
neutral; the 2e-5 continuation then drifts. This is the same pattern as the
local visited map, reward rebalance and player marker: the 99M policy is a
sharp local optimum, and 5M warm-start fine-tunes regress it. The tile map is
not ruled out by a from-scratch run, but the warm-start result is negative.

The next fair test is a clean from-scratch run with the final ruleset (legacy
observation, or legacy + local_tiles) on held-out seeds 10-19. The 99M policy
remains the deepest; gamma 0.999 is the speed/reliability alternative.

## Clean from-scratch run: final ruleset (development)

Started 2026-11-25, seed 43, 100M steps from scratch with no checkpoint:
legacy seven-component observation; level_weight 4.0, explore_weight 0.3;
61,440 horizon; gamma 0.995; LR 1e-4 until 78M then 5e-5; 32 envs. The
schedule is fixed before training; it is not a per-wave tuning knob.

Pre-registered selection: evaluate the 10M checkpoints on validation seeds
0-2 at 61,440; rank by Boulder reach, then Mt. Moon 3 Trainer 2 reach, then
Mt. Moon B2F reach, then lower Boulder median; evaluate the selected
checkpoint on held-out seeds 10-19. `scripts/clean_select.py` implements the
rule. The held-out report is the clean headline; the validation only picks
the checkpoint. Result pending.

## Clean from-scratch run: stopped at 24.1M (confirmation failed)

The clean 100M run (legacy observation; LR 1e-4 until 78M then 5e-5; seed 43)
was stopped at 24.1M because the pre-registered validation failed:

| checkpoint | Boulder | Viridian Forest | Pewter |
|---|---:|---:|---:|
| 5.33M badge snapshot | 0/3 | -- | -- |
| 5.79M best_model | 0/3 | -- | -- |
| 10M | 0/3 | 2/3 | 0/3 |
| 20M | 0/3 | 3/3 | 0/3 |
| 24.1M | 0/3 | -- | -- |

The 5.33M first badge was training-only. The 20M and 24.1M policies reached
Viridian Forest in every seed but never Pewter. At 24.1M the training window
was recovering (`badges_mean` 0.03 -> 0.31, Route 3 Trainer 6), but the
checkpoint did not reproduce it from a fresh start. The 5e-5 drop at 78M was
never reached.

The fixed 1e-4 from-scratch schedule does not reproduce the 99M policy. The
99M result is a path-dependent development lineage, not a clean from-scratch
result. The gamma-0.999 104M final remains the warm-start speed/reliability
alternative. No further warm-start or from-scratch run is planned without a
new method.

## Final state

The development program stops here. The **99M checkpoint** is the deepest
development policy: Boulder 10/10, Route 3 10/10, Mt. Moon 3 Trainer 2 9/10,
Mt. Moon B2F 10/10. The **gamma-0.999 104M final** is the warm-start
speed/reliability alternative: Boulder 10/10 at 5,160, Route 3 10/10,
Mt. Moon 3 t2 10/10, B2F 10/10. The full game was not reached: Trainer 3,
the Exit Super Nerd, and Cerulean remain 0/10.

The **clean from-scratch confirmation failed**: the 1e-4 schedule was stopped
at 24.1M with the 10M, 20M, and 24.1M checkpoints at **0/3 Boulder**. The 99M
is therefore a path-dependent development result, not a clean
sample-efficiency claim. The warm-start priors (local visited map, reward
rebalance, player marker, tile-ID map) all regressed; the only positive
method change was gamma 0.999.

What the project established:

- The map is the load-bearing observation: pixels-only fails, and the global
  map enables the early game.
- The coarse 48x48 map (~9x9 tiles per cell) is the indoor ceiling; minimal
  local maps and a player marker do not fix Mt. Moon.
- The 99M development lineage is not reproducible from scratch with the
  current fixed schedule; the clean run fails before Brock.
- The level/explore reward balance, local maps, marker, and tile map are not
  load-bearing at this stage; gamma 0.999 is a speed/reliability win.

Artifacts: `runs/best/red_99M_s43.zip` (deepest) and
`runs/best/red_gamma_104M_s43.zip` (speed). A future phase should test a
larger map prior (tile-level walkability) or a different algorithm from
scratch, not another warm-start.

## Phase 0: reopening the program

The development program above is frozen and released as v0.1.0. A new phase
starts from the same codebase and a new training lineage: no warm start, no
retuning of the 99M policy, and evaluation always from a cold `bulba_start`.

The first step is a pre-registered screen of why the clean from-scratch run
failed, before changing the method: four 20M single-axis arms (control, gamma
0.999, explore weight 3.0, level weight 1.0). The design, endpoints and
decision rules were frozen before the scored runs in
`docs/PREREGISTRATION_PHASE0.md`. Results will be appended here and in
`docs/PHASE0_RESULTS.md`.

The training callback now logs the reward mix (`game/reward_*_share`), battle
fraction, action entropy and per-episode map coverage, so "grinding" and
"policy collapse" are measured rather than inferred from the reward curve.

### 2026-11-26 02:38: resource crash on explore3

The baseline (231 min) and gamma (214 min) arms finished 20M. The third arm,
`phase0_explore3`, failed at environment startup with WinError 1455 ("paging
file is too small") and WinError 1450. The machine was at 54.5/55.4 GB commit
charge; ComfyUI alone held 11.1 GB. The 32 workers hung instead of exiting, so
the process tree was killed and `run_tiers.py` moved on to `phase0_level1`.

Recovery: ComfyUI, Discord and Steam were closed, restoring about 10 GB of
commit headroom. `level1` continued. A second pass
(`runs/phase0_sweep2.log`) waits for the first orchestrator to exit, then
reruns the crashed arm with the identical frozen config and fills any missing
evaluation reports. Pre-registration rule 6 permits rerunning a crashed arm
with the same settings; no arm setting changed.

The crash is an environment failure, not a result for the explore arm. The
arm has no checkpoints and is not part of any comparison until the rerun
finishes.

### Phase 0 result: no single knob reaches Boulder

The rerun completed `explore3`, and all eight fixed evaluations ran. Primary
endpoint (Boulder at 20M, three validation seeds):

| arm | 10M Boulder | 20M Boulder | 20M Forest | 20M Pewter | training badge |
|---|---:|---:|---:|---:|---:|
| baseline | 0/3 | **0/3** | 3/3 | 0/3 | 5.33M |
| gamma 0.999 | 0/3 | **0/3** | 0/3 | 0/3 | 13.62M |
| explore 3.0 | 0/3 | **0/3** | 3/3 | **2/3** | none |
| level 1.0 | 0/3 | **0/3** | 0/3 | 0/3 | 11.43M |

Pre-registration rule 2 applies: no single knob is sufficient. Phase 1
proceeds with the method change, with no fifth arm and no further reward
tuning for the first badge.

The secondary endpoints are still informative. `explore3` was the only arm to
reach Pewter from a cold start (2/3, median 32,120) and removed the level
grinding (level share 0.02, explore share 0.90), but it never earned a badge.
`level1` had the best 10M checkpoint (Forest 3/3, median 1,421) and regressed
by 20M, which points at retention rather than reward balance. Gamma 0.999,
positive under warm start, was the weakest from scratch. Action entropy stayed
1.89-1.96 in every arm, so none of these are policy collapses.

Phase 1 carries `explore_weight=3.0` and `gamma=0.995`, selects checkpoints on
validation instead of the final step, and adds the method change: recurrence,
the semantic local map, intrinsic exploration, and the self-generated archive.
Full write-up: `docs/PHASE0_RESULTS.md`.


