# KantoRL

![Version](https://img.shields.io/github/v/release/RoseOfficial/KantoRL?label=version)
![Downloads](https://img.shields.io/github/downloads/RoseOfficial/KantoRL/total)
![Lines of Code](https://aschey.tech/tokei/github/RoseOfficial/KantoRL?category=code)
![Code Size](https://img.shields.io/github/languages/code-size/RoseOfficial/KantoRL)
![Last Commit](https://img.shields.io/github/last-commit/RoseOfficial/KantoRL)
![Python](https://img.shields.io/github/languages/top/RoseOfficial/KantoRL)

**The MNIST of Pokemon Red for Reinforcement Learning**

A minimal, educational RL environment for Pokemon Red. Simple enough for students and hobbyists to understand and modify.

Measured results, and the harness bugs found while getting them, are recorded in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Results so far

Trained from `bulba_start` (Oak's Lab, level-6 Bulbasaur) with 32 PyBoy
environments. All evaluations are stochastic and start from the same save state.

| milestone | best result | where |
|---|---:|---|
| First Boulder Badge | 10.48M training steps | levelmax, first-seen log |
| Reliable badge | **10/10** (median 7,182 steps) | 99M best, 61k eval |
| Route 3, trainers 0-4 and 6-7 | 10/10 | 99M best |
| Route 3 trainer 5 | **90%** | 99M best |
| Mt. Moon 1 trainers 1, 2 and 4 | 10/10 | 99M best |
| Mt. Moon 3 trainer 2 | **90%** (median 28,482) | 99M best |
| Mt. Moon B2F | 10/10 (median 24,587) | 99M best |
| Exit Super Nerd / Cerulean City | not yet | — |

The current frontier is the **Mt. Moon B2F exit → Cerulean**. The full game has
not been completed: training episodes cap at 61,440 steps, and the current
policy reaches Mt. Moon B2F in every episode but has not beaten the Exit
Super Nerd. The 800M-step reference for a full Red clear is CJBoey's
stage-managed baseline; this project deliberately uses no stage manager, no
scripted HMs, and no route curriculum.

The 99M checkpoint was trained before a party-memory fix: levels and HP now use
the 44-byte party-struct stride and big-endian HP, and the old reads could
report fake level-89 party members. The checkpoint still scores 10/10 under the
corrected reads. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the 100M
collapse and the fix.

These numbers are a **development lineage**: cumulative steps with
human-in-the-loop horizon, learning-rate, observation and checkpoint changes.
Seeds 0-9 are used for development; seeds 10-19 are reserved for the final
fixed-ruleset measurement. The clean from-scratch run, not the best intermediate
checkpoint, is the headline sample-efficiency result.


## Setup

Verified on Windows with Python 3.13 and an 8-core/16-thread CPU.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
```

PyBoy ships SDL2 binaries in its wheel, so no separate emulator install is
needed. A ROM is *not* included; put yours somewhere outside the repo (the
`.gitignore` blocks `*.gb`, but a ROM should not live in a checkout at all) and
pass the path to every command.

Check that the ROM boots before spending time on training:

```powershell
.venv\Scripts\python -c "from pyboy import PyBoy; pb = PyBoy(r'C:\roms\Pokemon Red.gb', window='null'); print(pb.cartridge_title); pb.stop()"
```

Install the **CPU** build of PyTorch. `pip install torch` on Windows already
gives you this, which is what you want: at this model size a GPU is *slower*
(see [Throughput](#throughput)).

## Throughput

Two different numbers matter, and they disagree about the best configuration.
All figures below include `fast_mode=True` (PyBoy's optimised path, ~3.2x per
env) and the last-frame render that keeps screen observations live.

**Environment ceiling.** With `scripts/bench_throughput.py` (real `SubprocVecEnv`,
full observation, IPC included, no training loop):

| envs | steps/s | per-env steps/s | scaling | RSS |
|---:|---:|---:|---:|---:|
| 4 | 2,181 | 545 | 1.00x | 1.2 GB |
| 8 | 3,272 | 409 | 1.50x | 2.1 GB |
| 16 | 4,562 | 285 | 2.09x | 4.0 GB |
| 24 | 5,405 | 225 | 2.48x | 5.9 GB |
| 32 | 5,750 | 180 | 2.64x | 7.8 GB |
| 48 | 6,508 | 136 | 2.98x | 11.5 GB |

Memory is roughly 240 MB per environment. Per-env throughput falls steadily,
but total throughput keeps rising because each env is now cheap enough that the
policy process, not the emulators, is the limit.

**Training throughput.** That ceiling is not what training achieves. SB3
collects a rollout and then updates on the same CPU while every worker sits
idle, so the number to optimize is `scripts/bench_train.py`, which times the two
phases separately:

| envs | rollout threads | update threads | steps/s |
|---:|---:|---:|---:|
| 16 | 2 | 16 | 1,558 |
| 32 | 2 | 16 | **1,835** |
| 48 | 2 | 16 | 1,880 |
| 64 | 2 | 16 | 1,913 |

The update now consumes 44-47% of wall time, so doubling the environments buys
less than doubling the emulator speed did. 32 envs is the shipped setting:
64 is only 4% faster and costs twice the memory.

```powershell
.venv\Scripts\python scripts/bench_throughput.py "C:/roms/Pokemon Red.gb" --envs 4 8 16 32 48
.venv\Scripts\python scripts/bench_train.py "C:/roms/Pokemon Red.gb" --sweep
```

**`fast_mode` is a different emulator configuration, not just a speed knob.**
It skips sound and serial emulation and simplifies LCD timing, so runs made
with it are not step-comparable with earlier checkpoints. Treat it as a new
baseline.

Two things that sound like wins but are not: the rich progress bar costs
nothing measurable (48.0s vs 48.4s for 40k steps), and moving the policy to CUDA
made training 3x *slower* (292 steps/s) because per-step device sync dominates a
small CNN.

## Quick Start

```bash
# Install
pip install -e .

# Train an agent
kantorl train path/to/pokemon_red.gb

# Train with options
kantorl train pokemon_red.gb --envs 8 --steps 5000000 --session my_run

# Evaluate a checkpoint
kantorl eval runs/checkpoints/model_1000000.zip pokemon_red.gb --render

# Train with streaming to shared map (optional)
pip install websockets
kantorl train pokemon_red.gb --stream --stream-user "my-agent" --stream-color "#ff0033"
```

## Features

- **Minimal**: ~1,500 lines of code (not 50,000)
- **Educational**: Documented memory addresses and clean architecture
- **Compatible**: Works with stable-baselines3 and Gymnasium
- **Extensible**: Protocol-based reward system for custom objectives
- **Tiered**: Named observation, reward and horizon experiments in YAML
- **Retained**: First-seen milestone log plus best and per-badge checkpoints
- **Stream Visualization**: Optional real-time training visualization on shared map

## Benchmarks

Compare hyperparameters and find optimal settings:

```bash
# Install benchmark dependencies
pip install -e ".[benchmark]"

# Run a single benchmark (Bronze tier = first badge)
kantorl benchmark single pokemon_red.gb --tier bronze --max-steps 2000000

# Compare configurations from YAML file
kantorl benchmark run configs/benchmark.yaml pokemon_red.gb

# Optuna hyperparameter search
kantorl benchmark search pokemon_red.gb --trials 50 --tier bronze
```

**Milestone Tiers:**
- **Bronze**: 1 badge (fast iteration)
- **Silver**: 4 badges (mid-game validation)
- **Gold**: 8 badges (all gyms)
- **Champion**: Elite Four defeated

## Architecture

```
src/kantorl/
├── config.py      # Simple dataclass configuration
├── memory.py      # Documented memory reading
├── rewards.py     # Protocol-based reward system
├── obs/           # Composable observation producers
├── tiers/         # Named experiment definitions (YAML)
├── checkpoints.py # PPO / RecurrentPPO loading + tier inference
├── milestones.py  # First-step milestone evaluation
├── env.py         # Main Gymnasium environment
├── train.py       # PPO training script
├── callbacks.py   # Training callbacks
├── benchmarks/    # Benchmark comparison system
└── data/          # Map and event data
```

## Observation Space

The observation is composed from a configurable list of producers. The default
reproduces the original observation exactly; dropping components is how prior
ablations (tiers) are expressed:

```bash
# Pixels only — closest to tabula rasa
kantorl train pokemon_red.gb --obs screens

# Minimal progress signal
kantorl train pokemon_red.gb --obs screens health events
```

| Key | Space | What it gives the policy |
|---|---|---|
| `screens` | `Box(0, 255, (3, 72, 80))` | Stacked grayscale frames |
| `health` | `Box(0, 1, (1,))` | Party HP fraction |
| `level` | `Box(-1, 1, (8,))` | Fourier-encoded party levels |
| `badges` | `MultiBinary(8)` | Per-badge flags |
| `events` | `MultiBinary(2560)` | Full event-flag bitfield (progress oracle) |
| `map` | `Box(0, 255, (48, 48))` | Exploration bitmap |
| `local_map` | `Box(0, 255, (15, 15))` | Visited-tile window centered on the player (opt-in) |
| `recent_actions` | `MultiDiscrete([8]*3)` | Last three actions |

`local_map` is opt-in and not part of the legacy default. It is one cell per
tile rather than the global map's ~9x9 tiles per cell, so indoor rooms and
corridors keep their shape.

In Python:

```python
config = KantoConfig(rom_path="pokemon_red.gb", obs_components=("screens",))
env = KantoRedEnv(config=config)
```

Evaluation reconstructs the tier from the checkpoint's observation space, so a
pixels-only model loads without a shape mismatch.

## Tiers

A tier is a complete experiment definition: observation components, reward,
horizon and PPO settings. Tiers are ordered by how much hand-authored game
knowledge they assume, not by how well they score.

```bash
kantorl tier list
kantorl tier run t0_pure pokemon_red.gb --session runs/t0
```

| Tier | Observation | What varies | Assumes |
|---|---|---|---|
| `t0_pure` | `screens` | canonical ladder | action set + frame rate |
| `t1_minimal` | `screens`, `health` | canonical ladder | pixels + visible HP |
| `t2_pragmatic` | all seven | canonical ladder | RAM progress oracles |
| `t2_screens_map` | `screens`, `map` | ablation arm | pixels + spatial memory |
| `t2_all_no_map` | all but `map` | ablation arm | RAM, no spatial memory |
| `t2_level4` | all seven | sum-level potential, weight 4.0 | failed variant, documented |
| `t2_levelmax` | all seven | max-level potential, weight 4.0 | best early-game reward |
| `t2_levelmax_h40` | all seven | max-level, 40,960 horizon | failed alone, documented |
| `t2_levelmax_lr1e4` | all seven | max-level, 1e-4 | continuation tier |
| `t2_levelmax_h40_lr1e4` | all seven | max-level, 40,960, 1e-4 | current best tier |

The first three are the canonical ladder: they hold `max_steps`,
`frame_stacks`, `screen_size`, reward scale and `gamma` identical, so the only
difference is the observation. The rest are single-axis variants used to price
a reward, horizon or optimizer change. Curriculum is off in every tier because
it changes *where episodes start* — exactly the rule this project is trying to
avoid importing.

Tier definitions live in `src/kantorl/tiers/*.yaml`. Unknown keys are rejected
so a typo cannot silently change what you measured.

## Milestone Evaluation

Reward curves say what the reward function paid for. Milestones say when the
agent first reached something real in the game.

```bash
kantorl milestones runs/t0/checkpoints/model_1000000.zip pokemon_red.gb \
    --seeds 0 1 2 --budget 20000 --out runs/eval/t0
```

If training started from a save state, pass the same `--save-state` here.
Without it the policy is evaluated from a cold ROM boot — a starting state it
never trained on, which silently invalidates the comparison.

Every run uses the same seeds and step budget, so reports from different
checkpoints line up.

Note that `--seeds` is only replication under `--stochastic`. With the default
deterministic (argmax) policy, the emulator is deterministic too, so every seed
replays a byte-identical episode and the harness says so with a warning. Seed a
real spread with `--stochastic`; use one seed when you want a single
reproducible rollout. For each `(seed, episode, milestone)` the harness records
the first step at which the milestone was reached:

- `badge:N` — the Nth gym badge
- `event:I` — a curated story event bit (`data/events.json`)
- `map:M` — first visit to map `M`

Milestones are read from RAM **for evaluation only** and never fed to the
policy, so a pixels-only tier and a RAM-rich tier are measured by the same
ruler. Output:

- `milestones.json` — full report, episodes, catalog and summary
- `milestones_episodes.csv` — long format, one row per reached milestone
- `milestones_summary.csv` — reach rate and median first step per milestone

A milestone that was never reached is reported with `reach_rate: 0` and a null
median step, which is deliberately distinct from "reached at step 0".

### Retention

`FirstSeenCallback` writes `first_seen.csv` during training and snapshots the
policy when a milestone is first reached:

- `checkpoints/best_model.zip` — updated on each new event or badge.
- `checkpoints/badge_<n>_model.zip` — immutable snapshot at the first n-th badge.
- matching `_vecnormalize.pkl` and `_step.txt` files.

Dense cumulative checkpoints are pruned to the latest ~10M-step window plus
every 1M-step lineage mark; named snapshots and their normalizers are never
pruned. This keeps 1B steps at ~35 GB instead of ~320 GB.

`scripts/eval_runs.py --best` evaluates `best_model.zip` into `eval_best/`
without overwriting the latest-checkpoint report. The peak is worth keeping:
seed 43's 20M policy beats Brock 7/10, while the same run at 40M beats him 0/10.

The question the tiers ask is comparative, so `scripts/compare_milestones.py`
puts several reports in one table:

```bash
python scripts/compare_milestones.py runs/t0_pure/eval runs/t1_minimal/eval runs/t2_pragmatic/eval
```

Milestones reached at step 0 are marked `*`: they are already true before the
agent acts (the starting map, for example), so they carry no progress signal.

Checkpoints are self-describing: the loader reads the policy class from the
archive, so PPO and RecurrentPPO both work, and the observation tier, frame
stack and screen size are recovered from the recorded observation space.

## Action Space

```
0: NOOP   1: UP     2: DOWN   3: LEFT
4: RIGHT  5: A      6: B      7: START
```

## Reward System

Default reward (matching PokemonRedExperiments V2, with two corrections):

- **Event progress**: +4.0 per new event flag. Inherited starting flags are
  baselined and never paid.
- **Badge collection**: +5.0 per gym badge.
- **Level**: potential-based on the **strongest party member**, so catching a
  Pokémon pays 0 and a party wipe cannot re-earn levels. At the canonical
  `level_weight=1.0`: +0.5 per level below 15, +0.125 above. The levelmax
  tiers use weight 4.0: +2.0 and +0.5.
- **Map discovery**: +1.0 per new map ID.
- **Exploration**: +0.015 per new `(map, x, y)`. The set clears on each new
  event, so progress opens fresh ground.
- **Interaction**: +0.05 for the first A press at a coordinate, once per
  environment process, not once per episode.
- **Healing**: 0. Off because it was farmable: 78% of one run's reward while it
  made no progress.
- **Stuck penalty**: 0. Off because it also fires during dialogue and menus.

`VecNormalize(norm_reward=True, clip_reward=10.0)` rescales the summed reward
by the running return standard deviation and clips to ±10. The raw weights
above set relative preferences; the policy sees roughly unit-scale rewards.
There is no explicit time cost and no Champion completion bonus.

## Custom Rewards

```python
from kantorl.rewards import RewardFunction, GameState

class MyReward:
    def calculate(self, state: GameState, prev: GameState | None) -> float:
        return state.badges * 10.0  # Only reward badges

    def reset(self) -> None:
        pass

    def get_info(self) -> dict:
        return {}

env = KantoRedEnv(rom_path="pokemon_red.gb", reward_fn=MyReward())
```

## Requirements

- Python 3.10+
- Pokemon Red ROM (not included)
- PyBoy emulator (installed automatically)

## License

All rights reserved. See [LICENSE](LICENSE).
