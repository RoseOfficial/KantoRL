"""
Optuna hyperparameter search for KantoRL benchmarks.

This module provides Optuna integration for automated hyperparameter
optimization. It creates objective functions that run benchmark training
and uses badge progress for early pruning of underperforming trials.

Architecture Role:
    The Optuna search provides automated exploration of the hyperparameter
    space. It integrates with the benchmark runner to:
    1. Define search spaces for hyperparameters
    2. Create objective functions for optimization
    3. Enable early pruning based on intermediate badge progress
    4. Persist study results to SQLite for resumability

Usage:
    >>> from kantorl.benchmarks.optuna_search import create_study, run_search
    >>>
    >>> # Run hyperparameter search
    >>> study = run_search(
    ...     rom_path="pokemon_red.gb",
    ...     n_trials=50,
    ...     tier=MilestoneTier.BRONZE,
    ... )
    >>> print(f"Best params: {study.best_params}")

    # CLI usage:
    $ kantorl benchmark search --trials 50 --tier bronze

Dependencies:
    - optuna: Hyperparameter optimization framework
    - kantorl.benchmarks.runner: BenchmarkRunner for training
    - kantorl.benchmarks.scenarios: MilestoneTier definitions

References:
    - Optuna documentation: https://optuna.readthedocs.io/
    - Optuna pruning: https://optuna.readthedocs.io/en/stable/tutorial/pruning.html

Notes:
    - Optuna is an optional dependency (pip install optuna)
    - Studies are saved to SQLite for persistence across sessions
    - Early pruning dramatically speeds up search by abandoning bad trials
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kantorl.benchmarks.scenarios import MilestoneTier

if TYPE_CHECKING:
    import optuna


# =============================================================================
# OPTUNA AVAILABILITY CHECK
# =============================================================================


def _check_optuna_installed() -> None:
    """
    Check if Optuna is installed and raise helpful error if not.

    Raises:
        ImportError: If Optuna is not installed with installation instructions.
    """
    try:
        import optuna  # noqa: F401
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter search. "
            "Install it with: pip install optuna\n"
            "Or install with the benchmark extra: pip install kantorl[benchmark]"
        )


# =============================================================================
# DEFAULT SEARCH SPACE
# =============================================================================


DEFAULT_SEARCH_SPACE = {
    # Learning rate (log scale)
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    # Discount factor
    "gamma": {"type": "float", "low": 0.95, "high": 0.999},
    # Steps per rollout
    "n_steps": {"type": "categorical", "choices": [32, 64, 128, 256]},
    # Entropy coefficient (exploration bonus)
    "ent_coef": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
    # GAE lambda
    "gae_lambda": {"type": "float", "low": 0.9, "high": 0.99},
    # Clip range
    "clip_range": {"type": "float", "low": 0.1, "high": 0.3},
    # Reward scale
    "reward_scale": {"type": "float", "low": 0.1, "high": 1.0},
    # Explore weight
    "explore_weight": {"type": "float", "low": 0.01, "high": 0.5, "log": True},
}


def sample_hyperparameters(
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Sample hyperparameters from the search space using an Optuna trial.

    Args:
        trial: Optuna trial object for suggesting values.
        search_space: Custom search space dict. Uses DEFAULT_SEARCH_SPACE if None.
                     Format: {param_name: {"type": "float|int|categorical", ...}}

    Returns:
        Dict of sampled hyperparameter values.

    Example:
        >>> import optuna
        >>> study = optuna.create_study()
        >>> trial = study.ask()
        >>> params = sample_hyperparameters(trial)
        >>> print(params["learning_rate"])
    """
    search_space = search_space or DEFAULT_SEARCH_SPACE
    params: dict[str, Any] = {}

    for param_name, param_config in search_space.items():
        param_type = param_config["type"]

        if param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                param_config["low"],
                param_config["high"],
                log=param_config.get("log", False),
            )
        elif param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                param_config["low"],
                param_config["high"],
            )
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name,
                param_config["choices"],
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return params


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================


def create_objective(
    rom_path: str,
    tier: MilestoneTier = MilestoneTier.BRONZE,
    max_steps: int = 2_000_000,
    n_envs: int = 16,
    seed: int = 42,
    search_space: dict[str, dict[str, Any]] | None = None,
    output_dir: str | Path = "runs/optuna",
    save_state_path: str | None = None,
    verbose: int = 0,
):
    """
    Create an Optuna objective function for hyperparameter optimization.

    The objective function:
    1. Samples hyperparameters from the search space
    2. Runs a benchmark training trial with those parameters
    3. Returns steps to first badge (lower is better)
    4. Supports pruning based on intermediate badge progress

    Args:
        rom_path: Path to Pokemon Red ROM file.
        tier: Milestone tier to optimize for.
        max_steps: Maximum training steps per trial.
        n_envs: Number of parallel environments.
        seed: Random seed for reproducibility.
        search_space: Custom search space. Uses DEFAULT_SEARCH_SPACE if None.
        output_dir: Directory for trial outputs.
        save_state_path: Path to save state file (.state) for starting point.
        verbose: Verbosity level (0=silent, 1=progress).

    Returns:
        Callable objective function for optuna.study.optimize().

    Example:
        >>> objective = create_objective("pokemon_red.gb", tier=MilestoneTier.BRONZE)
        >>> study = optuna.create_study(direction="minimize")
        >>> study.optimize(objective, n_trials=50)
    """
    _check_optuna_installed()
    import optuna

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Runs a single benchmark trial and returns steps to first badge.
        Uses pruning callback to abandon underperforming trials early.
        """
        # Sample hyperparameters
        params = sample_hyperparameters(trial, search_space)

        if verbose:
            print(f"\n[Trial {trial.number}] Parameters: {params}")

        # Import here to avoid circular dependency
        from kantorl.benchmarks.runner import BenchmarkRunner

        # Create runner with sampled parameters
        runner = BenchmarkRunner(
            rom_path=rom_path,
            tier=tier,
            max_steps=max_steps,
            n_envs=n_envs,
            output_dir=Path(output_dir) / f"trial_{trial.number}",
            early_stop=True,
            save_state_path=save_state_path,
            verbose=verbose,
        )

        # Run training
        start_time = time.time()
        try:
            result = runner.run_single(
                config_name=f"trial_{trial.number}",
                config_overrides=params,
                seed=seed,
            )
        except Exception as e:
            # If training fails, return infinity (worst possible)
            if verbose:
                print(f"[Trial {trial.number}] Failed: {e}")
            return float("inf")

        elapsed = time.time() - start_time

        # Report intermediate values for pruning
        # Report badge progress at different step counts
        for badge, steps in result.steps_to_badge.items():
            trial.report(steps, step=badge)

            # Check if trial should be pruned
            if trial.should_prune():
                if verbose:
                    print(f"[Trial {trial.number}] Pruned at badge {badge}")
                raise optuna.TrialPruned()

        # Primary objective: steps to first badge
        steps_to_badge_1 = result.steps_to_badge.get(1, float("inf"))

        if verbose:
            if steps_to_badge_1 < float("inf"):
                print(
                    f"[Trial {trial.number}] Completed: {steps_to_badge_1:,.0f} steps "
                    f"to badge 1 ({elapsed:.1f}s)"
                )
            else:
                print(f"[Trial {trial.number}] No badge reached ({elapsed:.1f}s)")

        # Store additional metrics as user attributes
        trial.set_user_attr("final_badges", result.final_badges)
        trial.set_user_attr("final_events", result.final_events)
        trial.set_user_attr("total_steps", result.total_steps)
        trial.set_user_attr("wall_time", elapsed)

        return steps_to_badge_1

    return objective


# =============================================================================
# STUDY CREATION AND MANAGEMENT
# =============================================================================


def create_study(
    study_name: str = "kantorl_benchmark",
    storage: str | None = None,
    direction: str = "minimize",
    pruner: str = "median",
) -> optuna.Study:
    """
    Create an Optuna study for hyperparameter optimization.

    Args:
        study_name: Name of the study (for identification and persistence).
        storage: SQLite storage URL. If None, creates in-memory study.
                Example: "sqlite:///runs/optuna/study.db"
        direction: Optimization direction ("minimize" for steps, "maximize" for badges).
        pruner: Pruning strategy ("median", "percentile", or "none").

    Returns:
        Optuna Study object ready for optimization.

    Example:
        >>> study = create_study(
        ...     study_name="my_search",
        ...     storage="sqlite:///optuna.db",
        ... )
        >>> objective = create_objective("pokemon_red.gb")
        >>> study.optimize(objective, n_trials=50)
    """
    _check_optuna_installed()
    import optuna

    # Create pruner
    if pruner == "median":
        pruner_obj: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        )
    elif pruner == "percentile":
        pruner_obj = optuna.pruners.PercentilePruner(
            percentile=50,
            n_startup_trials=5,
        )
    elif pruner == "none":
        pruner_obj = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner}")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner_obj,
        load_if_exists=True,  # Resume if study already exists
    )

    return study


def run_search(
    rom_path: str,
    n_trials: int = 50,
    tier: MilestoneTier = MilestoneTier.BRONZE,
    max_steps: int = 2_000_000,
    n_envs: int = 16,
    seed: int = 42,
    search_space: dict[str, dict[str, Any]] | None = None,
    output_dir: str | Path = "runs/optuna",
    study_name: str = "kantorl_benchmark",
    storage: str | None = None,
    n_jobs: int = 1,
    save_state_path: str | None = None,
    verbose: int = 1,
) -> optuna.Study:
    """
    Run a complete hyperparameter search.

    Convenience function that creates a study and objective function,
    then runs the optimization loop.

    Args:
        rom_path: Path to Pokemon Red ROM file.
        n_trials: Number of trials to run.
        tier: Milestone tier to optimize for.
        max_steps: Maximum training steps per trial.
        n_envs: Number of parallel environments per trial.
        seed: Random seed for reproducibility.
        search_space: Custom search space. Uses DEFAULT_SEARCH_SPACE if None.
        output_dir: Directory for trial outputs.
        study_name: Name of the study.
        storage: SQLite storage URL for persistence. If None, uses in-memory.
        n_jobs: Number of parallel jobs (trials). 1 = sequential.
        save_state_path: Path to save state file (.state) for starting point.
        verbose: Verbosity level.

    Returns:
        Completed Optuna Study with results.

    Example:
        >>> study = run_search(
        ...     rom_path="pokemon_red.gb",
        ...     n_trials=50,
        ...     tier=MilestoneTier.BRONZE,
        ... )
        >>> print(f"Best params: {study.best_params}")
        >>> print(f"Best value: {study.best_value:,.0f} steps")
    """
    _check_optuna_installed()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default storage if not provided
    if storage is None:
        storage = f"sqlite:///{output_dir / 'study.db'}"

    # Create study
    study = create_study(
        study_name=study_name,
        storage=storage,
    )

    # Create objective
    objective = create_objective(
        rom_path=rom_path,
        tier=tier,
        max_steps=max_steps,
        n_envs=n_envs,
        seed=seed,
        search_space=search_space,
        output_dir=output_dir,
        save_state_path=save_state_path,
        verbose=verbose,
    )

    # Run optimization
    if verbose:
        print("\nStarting hyperparameter search:")
        print(f"  Trials: {n_trials}")
        print(f"  Tier: {tier.value}")
        print(f"  Max steps: {max_steps:,}")
        print(f"  Storage: {storage}")
        print()

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=verbose > 0,
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted. Results saved.")

    # Print summary
    if verbose and study.best_trial:
        print(f"\n{'='*60}")
        print("SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"  Best trial: {study.best_trial.number}")
        print(f"  Best value: {study.best_value:,.0f} steps to badge 1")
        print("  Best params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

    return study


def get_best_params(study: optuna.Study) -> dict[str, Any]:
    """
    Extract best parameters from a completed study.

    Args:
        study: Completed Optuna study.

    Returns:
        Dict of best hyperparameter values.
    """
    return study.best_params


def export_study_results(
    study: optuna.Study,
    output_path: str | Path,
) -> None:
    """
    Export study results to a CSV file.

    Args:
        study: Completed Optuna study.
        output_path: Path for output CSV file.
    """
    _check_optuna_installed()

    df = study.trials_dataframe()
    df.to_csv(output_path, index=False)
    print(f"Study results exported to: {output_path}")
