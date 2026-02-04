"""
Benchmarking system for KantoRL.

This module provides a standardized benchmark system for finding optimal RL settings
to beat Pokemon Red as fast as possible without scripting. It enables:

- **Reproducibility**: Validate same hyperparameters produce similar results
- **Regression detection**: Detect when code changes break training
- **Algorithm comparison**: Compare reward functions, observation spaces, RL settings
- **Community baselines**: Publish reference scores for comparison

Architecture Role:
    The benchmark system sits on top of the training pipeline and provides:
    1. Scenario definitions (Bronze/Silver/Gold/Champion milestones)
    2. Result tracking and metrics (BenchmarkResult dataclass)
    3. Training orchestration (BenchmarkRunner)
    4. Output formatting (Console/JSON/Markdown reporters)
    5. Config comparison and ranking (Comparator)
    6. Hyperparameter search (Optuna integration)

Primary Metric:
    Steps to reach milestone (lower is better). This measures how efficiently
    an agent learns to progress through the game.

Usage:
    # Run batch comparison of configs
    kantorl benchmark run config.yaml

    # Evaluate single checkpoint against milestones
    kantorl benchmark eval checkpoint.zip

    # Compare past benchmark results
    kantorl benchmark compare runs/benchmarks/

    # Optuna hyperparameter search
    kantorl benchmark search --trials 20

Module Structure:
    - scenarios.py: Milestone tier definitions (Bronze/Silver/Gold/Champion)
    - metrics.py: BenchmarkResult dataclass and metric calculations
    - runner.py: BenchmarkRunner for orchestrating training/eval runs
    - reporters.py: Console, JSON, and Markdown output formatters
    - comparator.py: Config comparison and ranking logic
    - optuna_search.py: Optional Optuna hyperparameter search integration

Dependencies:
    - kantorl.config: Configuration dataclass
    - kantorl.train: Training functionality
    - kantorl.eval: Evaluation functionality
    - stable-baselines3: PPO and vectorized environments
    - optuna (optional): Hyperparameter optimization

References:
    - Optuna: https://optuna.org/
    - stable-baselines3: https://stable-baselines3.readthedocs.io/
"""

# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

from kantorl.benchmarks.comparator import compare_results, rank_by_metric
from kantorl.benchmarks.metrics import BenchmarkResult
from kantorl.benchmarks.reporters import (
    ConsoleReporter,
    JSONReporter,
    MarkdownReporter,
)
from kantorl.benchmarks.runner import BenchmarkRunner
from kantorl.benchmarks.scenarios import (
    MilestoneTier,
    check_milestone_reached,
    get_tier_thresholds,
)

__all__ = [
    # Core types
    "BenchmarkResult",
    "MilestoneTier",
    # Functions
    "check_milestone_reached",
    "get_tier_thresholds",
    "compare_results",
    "rank_by_metric",
    # Classes
    "BenchmarkRunner",
    "ConsoleReporter",
    "JSONReporter",
    "MarkdownReporter",
]
