# KantoRL

![Version](https://img.shields.io/github/v/release/RoseOfficial/KantoRL?label=version)
![Downloads](https://img.shields.io/github/downloads/RoseOfficial/KantoRL/total)
![Lines of Code](https://aschey.tech/tokei/github/RoseOfficial/KantoRL?category=code)
![Code Size](https://img.shields.io/github/languages/code-size/RoseOfficial/KantoRL)
![Last Commit](https://img.shields.io/github/last-commit/RoseOfficial/KantoRL)
![Python](https://img.shields.io/github/languages/top/RoseOfficial/KantoRL)

**The MNIST of Pokemon Red for Reinforcement Learning**

A minimal, educational RL environment for Pokemon Red. Simple enough for students and hobbyists to understand and modify.

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
| `recent_actions` | `MultiDiscrete([8]*3)` | Last three actions |

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

| Tier | Observation | Reward | Assumes |
|---|---|---|---|
| `t0_pure` | `screens` | coordinate novelty | action set + frame rate |
| `t1_minimal` | `screens`, `health` | default (progress shaped) | pixels + visible HP |
| `t2_pragmatic` | all seven | default (progress shaped) | RAM progress oracles |

T0 and T1 and T2 hold `max_steps`, `frame_stacks`, `screen_size`, reward scale
and `gamma` identical. The pairwise difference is therefore the observation,
which is what makes the comparison an ablation rather than an anecdote.
Curriculum is off in all three because it changes *where episodes start*, a
separate axis; copy a tier file and set `enable_curriculum: true` to study it.

Tier definitions live in `src/kantorl/tiers/*.yaml`. Unknown keys are rejected
so a typo cannot silently change what you measured.

## Milestone Evaluation

Reward curves say what the reward function paid for. Milestones say when the
agent first reached something real in the game.

```bash
kantorl milestones runs/t0/checkpoints/model_1000000.zip pokemon_red.gb \
    --seeds 0 1 2 --budget 163840 --out runs/eval/t0
```

Every run uses the same seeds and step budget, so reports from different
checkpoints line up. For each `(seed, episode, milestone)` the harness records
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

Checkpoints are self-describing: the loader reads the policy class from the
archive, so PPO and RecurrentPPO both work, and the observation tier, frame
stack and screen size are recovered from the recorded observation space.

## Action Space

```
0: NOOP   1: UP     2: DOWN   3: LEFT
4: RIGHT  5: A      6: B      7: START
```

## Reward System

Default reward (matching PokemonRedExperiments V2):
- **Event progress**: +4.0 per new event triggered
- **Badge collection**: +5.0 per gym badge
- **Map discovery**: +1.0 per new map visited
- **Exploration**: +0.005 per new coordinate
- **Healing**: Proportional to HP restored
- **Stuck penalty**: -0.025 after 500 steps without progress

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
