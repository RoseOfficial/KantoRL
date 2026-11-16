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
