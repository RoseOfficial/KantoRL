"""
Benchmark result dataclass and metric calculations for KantoRL.

This module defines the BenchmarkResult dataclass that captures all metrics
from a benchmark run, including speed metrics, quality metrics, and efficiency
metrics. Results are designed to be easily serializable for JSON output and
comparable across different configurations.

Architecture Role:
    Metrics define HOW we measure. They provide:
    1. Standardized result structure (BenchmarkResult dataclass)
    2. Derived metric calculations (steps_per_badge, efficiency)
    3. Serialization support for JSON/Markdown output

    The runner (runner.py) creates BenchmarkResult instances during training,
    and the reporters (reporters.py) format them for output.

Primary Metric:
    steps_to_badge: Dict mapping badge number to steps taken to reach it.
    Lower values indicate faster learning and better sample efficiency.

Usage:
    >>> from kantorl.benchmarks.metrics import BenchmarkResult
    >>>
    >>> result = BenchmarkResult(
    ...     config_name="baseline",
    ...     steps_to_badge={1: 500000, 2: 1200000},
    ...     final_badges=2,
    ...     final_events=150,
    ...     seed=42,
    ... )
    >>> print(f"Steps per badge: {result.steps_per_badge:.0f}")

Dependencies:
    - dataclasses: For BenchmarkResult dataclass
    - typing: For type annotations
    - datetime: For timestamp handling
    - hashlib: For config hashing

References:
    - Dataclasses documentation: https://docs.python.org/3/library/dataclasses.html
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# =============================================================================
# BENCHMARK RESULT DATACLASS
# =============================================================================


@dataclass
class BenchmarkResult:
    """
    Complete result from a single benchmark run.

    This dataclass captures all metrics from training or evaluating an agent
    on Pokemon Red. It includes speed metrics (steps/time to milestones),
    quality metrics (final achievements), and efficiency metrics (derived ratios).

    The result is designed to be:
    - Serializable to JSON for machine-readable output
    - Comparable across different configurations and seeds
    - Rich enough for detailed analysis while staying minimal

    Attributes:
        config_name: Human-readable name for this configuration.
        seed: Random seed used for this run (for reproducibility).
        timestamp: ISO format timestamp when the run completed.

        steps_to_badge: Dict mapping badge number (1-8) to steps taken.
                        Only includes badges that were reached.
        wall_time_to_badge: Dict mapping badge number to wall-clock seconds.

        final_badges: Number of badges at end of training/eval.
        final_events: Number of events triggered at end.
        unique_maps_visited: Number of distinct map areas visited.
        total_healing: Cumulative HP restoration (Pokemon Center usage).

        total_steps: Total training steps taken.
        total_wall_time: Total wall-clock time in seconds.

        config_dict: Full configuration dictionary for reproducibility.
        config_hash: SHA256 hash of config for quick comparison.

    Example:
        >>> result = BenchmarkResult(
        ...     config_name="high_lr",
        ...     seed=42,
        ...     steps_to_badge={1: 500000, 2: 1200000},
        ...     final_badges=2,
        ...     final_events=150,
        ...     total_steps=2000000,
        ... )
        >>> print(result.steps_per_badge)  # Average steps per badge
        600000.0

    Notes:
        - steps_to_badge is the PRIMARY metric (lower is better)
        - Empty steps_to_badge means no badges were collected
        - Config hash allows grouping results with identical settings
    """

    # ===================
    # Identification
    # ===================

    # Human-readable name identifying this configuration
    # Example: "baseline", "high_lr", "aggressive_explore"
    config_name: str = "unnamed"

    # Random seed used for reproducibility
    # Different seeds with same config test statistical variance
    seed: int = 42

    # ISO format timestamp when run completed
    # Auto-generated if not provided
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # ===================
    # Speed Metrics (Primary - lower is better)
    # ===================

    # Steps taken to reach each badge
    # Key: badge number (1-8), Value: cumulative steps when badge was obtained
    # Example: {1: 500000, 2: 1200000} means badge 1 at 500K, badge 2 at 1.2M
    steps_to_badge: dict[int, int] = field(default_factory=dict)

    # Wall-clock time to reach each badge
    # Key: badge number (1-8), Value: seconds since training start
    wall_time_to_badge: dict[int, float] = field(default_factory=dict)

    # ===================
    # Quality Metrics
    # ===================

    # Final badge count at end of run (0-8)
    final_badges: int = 0

    # Final event flag count at end of run
    final_events: int = 0

    # Number of unique map areas visited
    unique_maps_visited: int = 0

    # Cumulative HP restoration (tracks Pokemon Center usage)
    total_healing: float = 0.0

    # ===================
    # Training Stats
    # ===================

    # Total steps taken during training
    total_steps: int = 0

    # Total wall-clock time in seconds
    total_wall_time: float = 0.0

    # ===================
    # Config Tracking
    # ===================

    # Full configuration dictionary for reproducibility
    # Contains all hyperparameters and settings
    config_dict: dict[str, Any] = field(default_factory=dict)

    # SHA256 hash of config for quick comparison
    # Auto-generated from config_dict if not provided
    config_hash: str = ""

    # ===================
    # Computed Properties
    # ===================

    @property
    def steps_per_badge(self) -> float:
        """
        Calculate average steps per badge collected.

        This efficiency metric shows how many steps it takes on average
        to earn each badge. Lower is better.

        Returns:
            Average steps per badge, or infinity if no badges collected.

        Example:
            >>> result = BenchmarkResult(steps_to_badge={1: 500000, 2: 1200000})
            >>> result.steps_per_badge
            600000.0
        """
        if self.final_badges == 0:
            return float("inf")
        return self.total_steps / self.final_badges

    @property
    def exploration_efficiency(self) -> float:
        """
        Calculate events triggered per 1000 steps.

        This metric measures how efficiently the agent triggers game events
        (story progress, items collected, etc.) relative to steps taken.
        Higher is better.

        Returns:
            Events per 1000 steps, or 0.0 if no steps taken.

        Example:
            >>> result = BenchmarkResult(final_events=150, total_steps=1000000)
            >>> result.exploration_efficiency
            0.15
        """
        if self.total_steps == 0:
            return 0.0
        return (self.final_events / self.total_steps) * 1000

    @property
    def badges_reached(self) -> list[int]:
        """
        Get list of badge numbers that were collected.

        Returns:
            Sorted list of badge numbers (1-8) that were reached.

        Example:
            >>> result = BenchmarkResult(steps_to_badge={1: 500000, 3: 2000000})
            >>> result.badges_reached
            [1, 3]
        """
        return sorted(self.steps_to_badge.keys())

    @property
    def first_badge_steps(self) -> int | None:
        """
        Get steps to first badge if collected.

        This is often the most important benchmark metric as it measures
        how quickly the agent can make initial progress.

        Returns:
            Steps to badge 1, or None if no badges collected.
        """
        return self.steps_to_badge.get(1)

    # ===================
    # Methods
    # ===================

    def __post_init__(self) -> None:
        """
        Validate and compute derived fields after initialization.

        - Generates config_hash from config_dict if not provided
        - Validates that steps_to_badge keys are in range 1-8
        """
        # Generate config hash if not provided
        if not self.config_hash and self.config_dict:
            self.config_hash = self._compute_config_hash()

        # Validate badge numbers are in valid range
        for badge_num in self.steps_to_badge:
            if not 1 <= badge_num <= 8:
                raise ValueError(f"Invalid badge number: {badge_num}. Must be 1-8.")

    def _compute_config_hash(self) -> str:
        """
        Compute SHA256 hash of configuration dictionary.

        The hash is computed from a JSON serialization of the config,
        ensuring consistent ordering.

        Returns:
            First 16 characters of SHA256 hex digest.
        """
        config_json = json.dumps(self.config_dict, sort_keys=True, default=str)
        full_hash = hashlib.sha256(config_json.encode()).hexdigest()
        return full_hash[:16]  # First 16 chars is sufficient for comparison

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of all fields and computed properties.

        Example:
            >>> result = BenchmarkResult(config_name="test", final_badges=2)
            >>> d = result.to_dict()
            >>> print(d["config_name"])
            test
        """
        return {
            # Identification
            "config_name": self.config_name,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            # Speed metrics
            "steps_to_badge": self.steps_to_badge,
            "wall_time_to_badge": self.wall_time_to_badge,
            # Quality metrics
            "final_badges": self.final_badges,
            "final_events": self.final_events,
            "unique_maps_visited": self.unique_maps_visited,
            "total_healing": self.total_healing,
            # Training stats
            "total_steps": self.total_steps,
            "total_wall_time": self.total_wall_time,
            # Computed metrics
            "steps_per_badge": self.steps_per_badge,
            "exploration_efficiency": self.exploration_efficiency,
            "badges_reached": self.badges_reached,
            # Full config (optional, may be large)
            "config_dict": self.config_dict,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkResult:
        """
        Create a BenchmarkResult from a dictionary.

        Useful for loading results from JSON files.

        Args:
            d: Dictionary containing result fields.
               Unknown keys are ignored for forward compatibility.

        Returns:
            BenchmarkResult instance.

        Example:
            >>> d = {"config_name": "test", "final_badges": 2, "seed": 42}
            >>> result = BenchmarkResult.from_dict(d)
            >>> print(result.config_name)
            test
        """
        # Get valid field names from dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}

        # Filter to only valid keys
        filtered = {k: v for k, v in d.items() if k in valid_keys}

        # Convert steps_to_badge keys to int (JSON serialization converts to str)
        if "steps_to_badge" in filtered:
            filtered["steps_to_badge"] = {
                int(k): v for k, v in filtered["steps_to_badge"].items()
            }
        if "wall_time_to_badge" in filtered:
            filtered["wall_time_to_badge"] = {
                int(k): v for k, v in filtered["wall_time_to_badge"].items()
            }

        return cls(**filtered)


# =============================================================================
# METRIC AGGREGATION FUNCTIONS
# =============================================================================


def aggregate_results(results: list[BenchmarkResult]) -> dict[str, Any]:
    """
    Aggregate multiple results (typically from different seeds) into statistics.

    Computes mean and standard deviation for key metrics across multiple runs
    of the same configuration with different seeds.

    Args:
        results: List of BenchmarkResult instances to aggregate.
                 Should have the same config but different seeds.

    Returns:
        Dictionary with aggregated statistics:
        - n_runs: Number of results aggregated
        - badges_mean, badges_std: Badge count statistics
        - events_mean, events_std: Event count statistics
        - steps_to_badge_mean: Dict of mean steps per badge
        - steps_to_badge_std: Dict of std dev per badge

    Example:
        >>> results = [result1, result2, result3]  # Same config, different seeds
        >>> stats = aggregate_results(results)
        >>> print(f"Badges: {stats['badges_mean']:.1f} +/- {stats['badges_std']:.1f}")

    Notes:
        - Returns empty stats if no results provided
        - Steps to badge only aggregates for badges reached by ALL runs
    """
    if not results:
        return {"n_runs": 0}

    import statistics

    # Basic counts
    badges = [r.final_badges for r in results]
    events = [r.final_events for r in results]

    # Steps to badge - only for badges reached by all runs
    all_badges_reached = set.intersection(*[set(r.badges_reached) for r in results])
    steps_to_badge_mean: dict[int, float] = {}
    steps_to_badge_std: dict[int, float] = {}

    for badge in sorted(all_badges_reached):
        badge_steps = [r.steps_to_badge[badge] for r in results]
        steps_to_badge_mean[badge] = statistics.mean(badge_steps)
        steps_to_badge_std[badge] = statistics.stdev(badge_steps) if len(badge_steps) > 1 else 0.0

    return {
        "n_runs": len(results),
        "config_name": results[0].config_name,
        "config_hash": results[0].config_hash,
        "badges_mean": statistics.mean(badges),
        "badges_std": statistics.stdev(badges) if len(badges) > 1 else 0.0,
        "events_mean": statistics.mean(events),
        "events_std": statistics.stdev(events) if len(events) > 1 else 0.0,
        "steps_to_badge_mean": steps_to_badge_mean,
        "steps_to_badge_std": steps_to_badge_std,
    }
