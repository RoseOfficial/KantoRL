"""
Benchmark runner for KantoRL.

This module provides the BenchmarkRunner class that orchestrates benchmark runs,
tracking milestone progress during training and producing BenchmarkResult instances.
It integrates with the existing train.py and callbacks.py infrastructure.

Architecture Role:
    The runner orchestrates WHEN and HOW benchmarks run. It provides:
    1. Training orchestration with milestone tracking
    2. Integration with stable-baselines3 callbacks
    3. Multi-seed support for statistical validity
    4. Configuration parsing and application

    The runner uses scenarios (scenarios.py) to define goals, metrics (metrics.py)
    to capture results, and reporters (reporters.py) to output findings.

Key Design Decisions:
    - Reuses existing train.py infrastructure (no duplicate code)
    - Uses callbacks for non-invasive milestone tracking
    - Supports both training (long) and evaluation (fast) modes
    - YAML config files for batch comparisons

Usage:
    >>> from kantorl.benchmarks import BenchmarkRunner, MilestoneTier
    >>>
    >>> runner = BenchmarkRunner(
    ...     rom_path="pokemon_red.gb",
    ...     tier=MilestoneTier.BRONZE,
    ...     max_steps=2_000_000,
    ... )
    >>> result = runner.run_single(config_name="baseline", seed=42)
    >>> print(f"Steps to badge 1: {result.first_badge_steps}")

Dependencies:
    - kantorl.config: KantoConfig dataclass
    - kantorl.train: Training infrastructure
    - kantorl.callbacks: Checkpoint and logging callbacks
    - stable_baselines3: PPO and vectorized environments
    - pyyaml: For config file parsing (optional)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from kantorl.benchmarks.metrics import BenchmarkResult
from kantorl.benchmarks.scenarios import (
    MilestoneTier,
    check_milestone_reached,
    get_tier_thresholds,
)
from kantorl.callbacks import (
    PerformanceCallback,
    StallDetectionCallback,
    TensorboardCallback,
)
from kantorl.config import KantoConfig
from kantorl.env import make_env

# =============================================================================
# BENCHMARK TRACKING CALLBACK
# =============================================================================


class BenchmarkTrackingCallback(BaseCallback):
    """
    Callback that tracks milestone progress during training.

    This callback monitors badge collection and event counts during training,
    recording the exact step and wall-clock time when each badge is obtained.
    It also checks for milestone completion to enable early stopping.

    Attributes:
        tier: The milestone tier we're benchmarking against.
        steps_to_badge: Dict mapping badge number to steps when obtained.
        wall_time_to_badge: Dict mapping badge number to seconds when obtained.
        start_time: Wall-clock time when training started.
        max_badges_seen: Highest badge count observed.
        max_events_seen: Highest event count observed.
        unique_maps: Set of map IDs visited.
        total_healing: Cumulative HP restoration.
        milestone_reached: Whether the target milestone has been reached.

    Example:
        >>> callback = BenchmarkTrackingCallback(tier=MilestoneTier.BRONZE)
        >>> model.learn(total_timesteps=1000000, callback=callback)
        >>> print(f"Steps to badge 1: {callback.steps_to_badge.get(1)}")

    Notes:
        - Records first time each badge is obtained (not re-obtainments)
        - Early stopping can be disabled by setting tier to None
        - Integrates with stable-baselines3's callback system
    """

    def __init__(
        self,
        tier: MilestoneTier | None = None,
        early_stop: bool = True,
        verbose: int = 1,
    ):
        """
        Initialize the benchmark tracking callback.

        Args:
            tier: Milestone tier to track against. None disables milestone checking.
            early_stop: Whether to stop training when milestone is reached.
            verbose: Verbosity level. 1=print when badges are obtained.
        """
        super().__init__(verbose)
        self.tier = tier
        self.early_stop = early_stop

        # Tracking state
        self.steps_to_badge: dict[int, int] = {}
        self.wall_time_to_badge: dict[int, float] = {}
        self.start_time: float = 0.0
        self.max_badges_seen: int = 0
        self.max_events_seen: int = 0
        self.unique_maps: set[int] = set()
        self.total_healing: float = 0.0
        self.milestone_reached: bool = False

    def _on_training_start(self) -> None:
        """Called when training starts. Records start time."""
        self.start_time = time.time()

    def _on_step(self) -> bool:
        """
        Called after each training step. Tracks badges and checks milestone.

        Returns:
            True to continue training, False to stop (if milestone reached
            and early_stop is enabled).
        """
        # Get infos from all environments
        for info in self.locals.get("infos", []):
            # Track badges
            badges = info.get("badges", 0)
            if badges > self.max_badges_seen:
                # New badge(s) obtained
                for badge_num in range(self.max_badges_seen + 1, badges + 1):
                    elapsed = time.time() - self.start_time
                    self.steps_to_badge[badge_num] = self.num_timesteps
                    self.wall_time_to_badge[badge_num] = elapsed

                    if self.verbose:
                        print(
                            f"Badge {badge_num} obtained at step {self.num_timesteps:,} "
                            f"({elapsed:.1f}s)"
                        )

                self.max_badges_seen = badges

            # Track events
            events = info.get("events", 0)
            if events > self.max_events_seen:
                self.max_events_seen = events

            # Track maps visited
            if "map_id" in info:
                self.unique_maps.add(info["map_id"])

            # Track healing
            if "total_healing" in info:
                self.total_healing = max(self.total_healing, info["total_healing"])

        # Check milestone
        if self.tier is not None and not self.milestone_reached:
            if check_milestone_reached(self.max_badges_seen, self.max_events_seen, self.tier):
                self.milestone_reached = True
                if self.verbose:
                    thresholds = get_tier_thresholds(self.tier)
                    print(
                        f"Milestone {self.tier.value.upper()} reached! "
                        f"({thresholds.badges} badges, {thresholds.events}+ events)"
                    )
                if self.early_stop:
                    return False  # Stop training

        return True  # Continue training

    def get_result_data(self) -> dict[str, Any]:
        """
        Get tracking data for BenchmarkResult construction.

        Returns:
            Dictionary with all tracked metrics.
        """
        return {
            "steps_to_badge": self.steps_to_badge.copy(),
            "wall_time_to_badge": self.wall_time_to_badge.copy(),
            "final_badges": self.max_badges_seen,
            "final_events": self.max_events_seen,
            "unique_maps_visited": len(self.unique_maps),
            "total_healing": self.total_healing,
            "total_wall_time": time.time() - self.start_time if self.start_time else 0.0,
        }


# =============================================================================
# BENCHMARK PROGRESS CALLBACK
# =============================================================================


class BenchmarkProgressCallback(BaseCallback):
    """
    Compact console progress callback for benchmark runs.

    Prints a single summary line every ``log_freq`` steps showing the best
    badges, events, and exploration tiles observed so far, plus the current
    mean step reward and throughput (steps/sec).

    Output format::

        [50K] badges:0 events:12 tiles:45 | rew:1.23 | 1,456 sps
        [100K] badges:0 events:28 tiles:89 | rew:2.10 | 1,502 sps

    This fills the gap between per-metric TensorBoard callbacks and the sparse
    badge milestone prints from :class:`BenchmarkTrackingCallback`.

    Attributes:
        log_freq: Print a progress line every N training steps.
        max_badges: Best badge count seen across all envs (cumulative).
        max_events: Best event count seen across all envs (cumulative).
        max_tiles: Best exploration tile count seen across all envs (cumulative).
        start_time: Wall-clock time when training started.
        last_log_time: Timestamp of the last progress line (for sps calc).
        last_log_steps: Step count at the last progress line.

    Example:
        >>> callback = BenchmarkProgressCallback(log_freq=50_000, verbose=1)
        >>> model.learn(total_timesteps=500_000, callback=callback)
        [50K] badges:0 events:8 tiles:34 | rew:0.45 | 1,523 sps

    Notes:
        - Metrics are cumulative maximums (best ever, not windowed).
        - Mean reward is a rolling average of per-step rewards collected
          directly from the training loop (no Monitor wrapper needed).
        - This callback does NOT log to TensorBoard — use TensorboardCallback
          for that.
    """

    def __init__(self, log_freq: int = 50_000, verbose: int = 1):
        """
        Initialize the benchmark progress callback.

        Args:
            log_freq: Print a progress summary every N training steps.
                     50,000 is a good balance between visibility and spam.
            verbose: Verbosity level. 1=print progress lines to console.
        """
        super().__init__(verbose)
        self.log_freq = log_freq

        # Cumulative best-ever metrics across all envs
        self.max_badges: int = 0
        self.max_events: int = 0
        self.max_tiles: int = 0

        # Rolling window of recent step rewards for mean reward display.
        # Collects per-env rewards each step; 10K entries covers ~1K steps
        # with 8 envs, giving a smooth rolling average.
        self._recent_rewards: deque[float] = deque(maxlen=10_000)

        # Timing state for steps-per-second calculation
        self.start_time: float = 0.0
        self.last_log_time: float = 0.0
        self.last_log_steps: int = 0

    def _on_training_start(self) -> None:
        """Called when training starts. Records start time for sps calculation."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_steps = 0

    def _on_step(self) -> bool:
        """
        Called after each training step. Tracks maximums and prints progress.

        Collects best-ever badge/event/tile counts from env infos, then
        prints a compact summary at each ``log_freq`` interval.

        Returns:
            True to continue training (never stops training).
        """
        # Update cumulative maximums from all environments
        for info in self.locals.get("infos", []):
            self.max_badges = max(self.max_badges, info.get("badges", 0))
            self.max_events = max(self.max_events, info.get("events", 0))
            self.max_tiles = max(self.max_tiles, info.get("unique_coords", 0))

        # Collect per-step rewards directly from the training loop.
        # self.locals["rewards"] is a numpy array of shape (n_envs,) containing
        # the reward each env returned this step. This avoids depending on
        # SB3's ep_info_buffer (which requires a Monitor wrapper).
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._recent_rewards.extend(rewards.tolist())

        # Print progress at regular intervals
        if self.num_timesteps % self.log_freq == 0 and self.verbose:
            # Format step count as compact string (e.g., 50K, 1.05M)
            step_str = _format_steps(self.num_timesteps)

            # Mean reward from rolling window of recent step rewards
            rew_str = "---"
            if self._recent_rewards:
                rew_str = f"{np.mean(self._recent_rewards):.2f}"

            # Calculate steps/sec since last log
            now = time.time()
            elapsed = now - self.last_log_time
            steps_done = self.num_timesteps - self.last_log_steps
            sps = int(steps_done / max(elapsed, 1e-6))

            print(
                f"[{step_str}] badges:{self.max_badges} "
                f"events:{self.max_events} tiles:{self.max_tiles} "
                f"| rew:{rew_str} | {sps:,} sps"
            )

            # Update timing state
            self.last_log_time = now
            self.last_log_steps = self.num_timesteps

        return True


def _format_steps(steps: int) -> str:
    """
    Format a step count as a compact human-readable string.

    Uses integer arithmetic to pick the right precision so that labels
    at 50K intervals are always unique (no duplicate rounding).

    Args:
        steps: Number of training steps.

    Returns:
        Compact string like "50K", "1.05M", or "100" for small values.

    Examples:
        >>> _format_steps(50_000)
        '50K'
        >>> _format_steps(1_050_000)
        '1.05M'
        >>> _format_steps(1_500_000)
        '1.5M'
        >>> _format_steps(2_000_000)
        '2M'
        >>> _format_steps(100)
        '100'
    """
    if steps >= 1_000_000:
        # Use integer arithmetic to avoid floating-point rounding ambiguity
        if steps % 1_000_000 == 0:
            return f"{steps // 1_000_000}M"
        elif steps % 100_000 == 0:
            # Clean tenths: 1.1M, 1.5M, etc.
            return f"{steps / 1_000_000:.1f}M"
        else:
            # Finer intervals (e.g., 50K): 1.05M, 1.15M, etc.
            return f"{steps / 1_000_000:.2f}M"
    elif steps >= 1_000:
        if steps % 1_000 == 0:
            return f"{steps // 1_000}K"
        else:
            return f"{steps / 1_000:.1f}K"
    return str(steps)


# =============================================================================
# BENCHMARK CONFIGURATION
# =============================================================================


@dataclass
class BenchmarkConfig:
    """
    Configuration for a benchmark run.

    Bundles all settings needed to run a benchmark, separate from KantoConfig
    which handles environment/training settings.

    Attributes:
        tier: Milestone tier to benchmark against.
        max_steps: Maximum training steps (benchmark will stop at milestone or this).
        n_envs: Number of parallel environments.
        seeds: List of seeds to run (for statistical validity).
        eval_episodes: Number of episodes for evaluation mode.
        early_stop: Whether to stop when milestone is reached.
        save_state_path: Optional explicit path to save state file. If None,
            the environment will auto-detect common state files next to the ROM.

    Example:
        >>> config = BenchmarkConfig(
        ...     tier=MilestoneTier.BRONZE,
        ...     max_steps=2_000_000,
        ...     seeds=[42, 123, 456],
        ... )
    """

    tier: MilestoneTier = MilestoneTier.BRONZE
    max_steps: int = 2_000_000
    n_envs: int = 16
    seeds: list[int] = field(default_factory=lambda: [42])
    eval_episodes: int = 10
    early_stop: bool = True
    save_state_path: str | None = None

    # Streaming configuration for real-time visualization
    enable_streaming: bool = False
    stream_username: str = "kantorl-bench"
    stream_color: str = "#ff0000"
    stream_sprite_id: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkConfig:
        """Create from dictionary, handling tier conversion."""
        d = d.copy()
        if "tier" in d and isinstance(d["tier"], str):
            d["tier"] = MilestoneTier(d["tier"])
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


class BenchmarkRunner:
    """
    Orchestrates benchmark runs for comparing RL configurations.

    The runner handles:
    - Creating environments with specified configurations
    - Running training with milestone tracking
    - Collecting results into BenchmarkResult instances
    - Supporting multiple seeds for statistical validity

    Attributes:
        rom_path: Path to Pokemon Red ROM file.
        benchmark_config: Benchmark settings (tier, max_steps, etc.).
        output_dir: Directory for benchmark outputs.
        verbose: Verbosity level for progress output.

    Example:
        >>> runner = BenchmarkRunner(
        ...     rom_path="pokemon_red.gb",
        ...     tier=MilestoneTier.BRONZE,
        ...     max_steps=2_000_000,
        ... )
        >>>
        >>> # Run single configuration
        >>> result = runner.run_single(config_name="baseline", seed=42)
        >>>
        >>> # Run with multiple seeds
        >>> results = runner.run_with_seeds(config_name="baseline", seeds=[42, 123])

    Notes:
        - Results are returned as BenchmarkResult instances
        - Training stops early if milestone is reached (configurable)
        - Uses SubprocVecEnv for parallel training
    """

    def __init__(
        self,
        rom_path: str,
        tier: MilestoneTier = MilestoneTier.BRONZE,
        max_steps: int = 2_000_000,
        n_envs: int = 16,
        output_dir: str | Path = "runs/benchmarks",
        early_stop: bool = True,
        save_state_path: str | None = None,
        enable_streaming: bool = False,
        stream_username: str = "kantorl-bench",
        stream_color: str = "#ff0000",
        stream_sprite_id: int = 0,
        verbose: int = 1,
    ):
        """
        Initialize the benchmark runner.

        Args:
            rom_path: Path to Pokemon Red ROM file (.gb).
            tier: Milestone tier to benchmark against.
            max_steps: Maximum training steps.
            n_envs: Number of parallel environments.
            output_dir: Directory for benchmark outputs.
            early_stop: Stop training when milestone is reached.
            save_state_path: Path to save state file (.state). If None, the
                environment auto-detects common state files next to the ROM.
            enable_streaming: Enable WebSocket streaming for visualization.
            stream_username: Display name on the shared map.
            stream_color: Hex color code for map marker.
            stream_sprite_id: Character sprite ID (0-50) for map display.
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        """
        self.rom_path = rom_path
        self.benchmark_config = BenchmarkConfig(
            tier=tier,
            max_steps=max_steps,
            n_envs=n_envs,
            early_stop=early_stop,
            save_state_path=save_state_path,
            enable_streaming=enable_streaming,
            stream_username=stream_username,
            stream_color=stream_color,
            stream_sprite_id=stream_sprite_id,
        )
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single(
        self,
        config_name: str = "default",
        config_overrides: dict[str, Any] | None = None,
        seed: int = 42,
        reward_fn: str = "default",
    ) -> BenchmarkResult:
        """
        Run a single benchmark training run.

        Creates environments, runs PPO training with milestone tracking,
        and returns a BenchmarkResult with all metrics.

        Args:
            config_name: Human-readable name for this configuration.
            config_overrides: Dict of KantoConfig overrides (e.g., learning_rate).
            seed: Random seed for reproducibility.
            reward_fn: Reward function name ("default", "badges_only", "exploration").

        Returns:
            BenchmarkResult with all tracked metrics.

        Example:
            >>> result = runner.run_single(
            ...     config_name="high_lr",
            ...     config_overrides={"learning_rate": 1e-3},
            ...     seed=42,
            ... )
        """
        config_overrides = config_overrides or {}

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running benchmark: {config_name} (seed={seed})")
            print(f"Tier: {self.benchmark_config.tier.value}")
            print(f"Max steps: {self.benchmark_config.max_steps:,}")
            print(f"{'='*60}\n")

        # Create KantoConfig with overrides (includes streaming settings)
        kanto_config = KantoConfig(
            rom_path=self.rom_path,
            save_state_path=self.benchmark_config.save_state_path,
            session_path=self.output_dir / config_name / f"seed_{seed}",
            n_envs=self.benchmark_config.n_envs,
            enable_streaming=self.benchmark_config.enable_streaming,
            stream_username=self.benchmark_config.stream_username,
            stream_color=self.benchmark_config.stream_color,
            stream_sprite_id=self.benchmark_config.stream_sprite_id,
            **config_overrides,
        )

        # Create vectorized environment
        env_fns = [
            make_env(
                self.rom_path,
                kanto_config,
                rank=i,
                seed=seed,
                reward_fn=reward_fn,
                enable_streaming=self.benchmark_config.enable_streaming,
            )
            for i in range(self.benchmark_config.n_envs)
        ]
        env = SubprocVecEnv(env_fns)

        try:
            # Create tracking callback
            tracking_callback = BenchmarkTrackingCallback(
                tier=self.benchmark_config.tier,
                early_stop=self.benchmark_config.early_stop,
                verbose=self.verbose,
            )

            # Create PPO model with TensorBoard logging enabled
            tb_path = kanto_config.session_path / "tensorboard"
            model = PPO(
                "MultiInputPolicy",
                env,
                n_steps=kanto_config.n_steps,
                batch_size=kanto_config.batch_size,
                n_epochs=kanto_config.n_epochs,
                gamma=kanto_config.gamma,
                gae_lambda=kanto_config.gae_lambda,
                clip_range=kanto_config.clip_range,
                ent_coef=kanto_config.ent_coef,
                vf_coef=kanto_config.vf_coef,
                learning_rate=kanto_config.learning_rate,
                tensorboard_log=str(tb_path),
                verbose=0 if self.verbose < 2 else 1,
                seed=seed,
            )

            # Compose callback stack:
            #   tracking_callback  - badge milestone timing (existing)
            #   TensorboardCallback - game metrics (badges/events/tiles) to TB
            #   PerformanceCallback - steps/sec to TB
            #   StallDetectionCallback - console warnings when stuck
            #   BenchmarkProgressCallback - compact console summary line
            callbacks = CallbackList([
                tracking_callback,
                TensorboardCallback(log_freq=1000, verbose=0),
                PerformanceCallback(log_freq=10_000, verbose=0),
                StallDetectionCallback(check_freq=50_000, verbose=1),
                BenchmarkProgressCallback(log_freq=50_000, verbose=self.verbose),
            ])

            # Run training
            model.learn(
                total_timesteps=self.benchmark_config.max_steps,
                callback=callbacks,
                progress_bar=self.verbose > 0,
            )

            # Build result (tracking_data includes total_wall_time from callback)
            tracking_data = tracking_callback.get_result_data()
            result = BenchmarkResult(
                config_name=config_name,
                seed=seed,
                total_steps=model.num_timesteps,
                config_dict={**config_overrides, "reward_fn": reward_fn},
                **tracking_data,
            )

            if self.verbose:
                print(f"\nBenchmark complete: {config_name} (seed={seed})")
                print(f"  Final badges: {result.final_badges}")
                print(f"  Final events: {result.final_events}")
                print(f"  Total steps: {result.total_steps:,}")
                print(f"  Wall time: {result.total_wall_time:.1f}s")

            return result

        finally:
            env.close()

    def run_with_seeds(
        self,
        config_name: str = "default",
        config_overrides: dict[str, Any] | None = None,
        seeds: list[int] | None = None,
        reward_fn: str = "default",
    ) -> list[BenchmarkResult]:
        """
        Run benchmark with multiple seeds for statistical validity.

        Runs the same configuration multiple times with different seeds to
        measure variance and compute confidence intervals.

        Args:
            config_name: Human-readable name for this configuration.
            config_overrides: Dict of KantoConfig overrides.
            seeds: List of seeds to use. Defaults to benchmark config seeds.
            reward_fn: Reward function name.

        Returns:
            List of BenchmarkResult instances, one per seed.

        Example:
            >>> results = runner.run_with_seeds(
            ...     config_name="baseline",
            ...     seeds=[42, 123, 456],
            ... )
            >>> mean_badges = np.mean([r.final_badges for r in results])
        """
        seeds = seeds or self.benchmark_config.seeds

        results = []
        for i, seed in enumerate(seeds):
            if self.verbose:
                print(f"\n[{i+1}/{len(seeds)}] Running seed {seed}...")

            result = self.run_single(
                config_name=config_name,
                config_overrides=config_overrides,
                seed=seed,
                reward_fn=reward_fn,
            )
            results.append(result)

        return results

    def run_comparison(
        self,
        configs: dict[str, dict[str, Any]],
        seeds: list[int] | None = None,
        reward_fn: str = "default",
    ) -> dict[str, list[BenchmarkResult]]:
        """
        Run benchmark comparison across multiple configurations.

        Runs each configuration with multiple seeds and returns results
        organized by configuration name.

        Args:
            configs: Dict mapping config names to override dicts.
                    Example: {"baseline": {}, "high_lr": {"learning_rate": 1e-3}}
            seeds: List of seeds for each config. Defaults to benchmark config seeds.
            reward_fn: Reward function name.

        Returns:
            Dict mapping config names to lists of BenchmarkResult instances.

        Example:
            >>> configs = {
            ...     "baseline": {},
            ...     "high_lr": {"learning_rate": 1e-3},
            ...     "low_lr": {"learning_rate": 1e-5},
            ... }
            >>> all_results = runner.run_comparison(configs, seeds=[42, 123])
        """
        all_results: dict[str, list[BenchmarkResult]] = {}

        for config_name, config_overrides in configs.items():
            if self.verbose:
                print(f"\n{'#'*60}")
                print(f"# Configuration: {config_name}")
                print(f"{'#'*60}")

            results = self.run_with_seeds(
                config_name=config_name,
                config_overrides=config_overrides,
                seeds=seeds,
                reward_fn=reward_fn,
            )
            all_results[config_name] = results

        return all_results


# =============================================================================
# YAML CONFIG PARSING
# =============================================================================


def load_benchmark_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load benchmark configuration from a YAML file.

    The YAML file should have the following structure:
    ```yaml
    benchmark:
      tier: bronze
      max_steps: 2_000_000
      seeds: [42, 123, 456]

    configs:
      baseline: {}
      high_lr:
        learning_rate: 1e-3
    ```

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If config file doesn't exist.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config file loading. "
            "Install it with: pip install pyyaml"
        )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def run_from_config(
    config_path: str | Path,
    rom_path: str,
    output_dir: str | Path = "runs/benchmarks",
    verbose: int = 1,
) -> dict[str, list[BenchmarkResult]]:
    """
    Run benchmarks from a YAML configuration file.

    Convenience function that loads a config file and runs all specified
    configurations with the specified seeds.

    Args:
        config_path: Path to YAML configuration file.
        rom_path: Path to Pokemon Red ROM file.
        output_dir: Directory for benchmark outputs.
        verbose: Verbosity level.

    Returns:
        Dict mapping config names to lists of BenchmarkResult instances.

    Example:
        >>> results = run_from_config("benchmark_config.yaml", "pokemon_red.gb")
    """
    config = load_benchmark_config(config_path)

    # Extract benchmark settings
    benchmark_settings = config.get("benchmark", {})
    tier_str = benchmark_settings.get("tier", "bronze")
    tier = MilestoneTier(tier_str)
    max_steps = benchmark_settings.get("max_steps", 2_000_000)
    n_envs = benchmark_settings.get("n_envs", 16)
    seeds = benchmark_settings.get("seeds", [42])
    early_stop = benchmark_settings.get("early_stop", True)
    save_state_path = benchmark_settings.get("save_state_path")

    # Extract configs to compare
    configs = config.get("configs", {"default": {}})

    # Create runner
    runner = BenchmarkRunner(
        rom_path=rom_path,
        tier=tier,
        max_steps=max_steps,
        n_envs=n_envs,
        output_dir=output_dir,
        early_stop=early_stop,
        save_state_path=save_state_path,
        verbose=verbose,
    )

    # Run comparison
    return runner.run_comparison(configs=configs, seeds=seeds)
