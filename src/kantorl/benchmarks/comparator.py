"""
Benchmark result comparison utilities for KantoRL.

This module provides functions for comparing and ranking benchmark results
across different configurations. It enables identifying which hyperparameters
or reward functions lead to faster training.

Architecture Role:
    The comparator analyzes benchmark results to answer:
    - Which configuration reaches milestones fastest?
    - How much improvement does each change provide?
    - Are differences statistically significant?

    Results from the runner (runner.py) are passed through the comparator
    before being sent to reporters (reporters.py).

Usage:
    >>> from kantorl.benchmarks.comparator import compare_results, rank_by_metric
    >>>
    >>> # Rank configs by steps to badge 1
    >>> rankings = rank_by_metric(results, badge=1)
    >>> winner = rankings[0]
    >>>
    >>> # Compare two specific configs
    >>> comparison = compare_results(results["baseline"], results["high_lr"])
    >>> print(f"Improvement: {comparison['improvement_pct']:.1f}%")

Dependencies:
    - statistics: For mean, stdev calculations
    - typing: For type annotations

References:
    - Statistical testing: https://docs.python.org/3/library/statistics.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kantorl.benchmarks.metrics import BenchmarkResult, aggregate_results

# =============================================================================
# COMPARISON RESULT
# =============================================================================


@dataclass
class ComparisonResult:
    """
    Result of comparing two benchmark configurations.

    Attributes:
        config_a: Name of first configuration.
        config_b: Name of second configuration.
        metric: The metric being compared.
        value_a: Metric value for config A.
        value_b: Metric value for config B.
        difference: Absolute difference (B - A).
        improvement_pct: Percentage improvement of B over A.
                        Positive means B is better (for "lower is better" metrics).
        winner: Name of winning configuration, or "tie" if negligible difference.

    Example:
        >>> comparison = ComparisonResult(
        ...     config_a="baseline",
        ...     config_b="high_lr",
        ...     metric="steps_to_badge_1",
        ...     value_a=500000,
        ...     value_b=400000,
        ...     difference=-100000,
        ...     improvement_pct=20.0,
        ...     winner="high_lr",
        ... )
    """

    config_a: str
    config_b: str
    metric: str
    value_a: float
    value_b: float
    difference: float
    improvement_pct: float
    winner: str


# =============================================================================
# RANKING FUNCTIONS
# =============================================================================


def rank_by_metric(
    results: dict[str, list[BenchmarkResult]],
    badge: int = 1,
    metric: str = "steps_to_badge",
) -> list[tuple[str, float, float]]:
    """
    Rank configurations by a metric (lower is better for steps metrics).

    Args:
        results: Dict mapping config names to lists of BenchmarkResult.
        badge: Badge number to rank by (1-8) when using steps_to_badge.
        metric: Metric to rank by. Options:
                - "steps_to_badge": Steps to reach specified badge (default)
                - "final_badges": Final badge count (higher is better)
                - "final_events": Final event count (higher is better)

    Returns:
        List of tuples (config_name, mean_value, std_dev), sorted by mean.
        For steps_to_badge, sorted ascending (lower is better).
        For badges/events, sorted descending (higher is better).

    Example:
        >>> rankings = rank_by_metric(results, badge=1)
        >>> print(f"Winner: {rankings[0][0]} with {rankings[0][1]:,.0f} steps")
    """
    rankings: list[tuple[str, float, float]] = []

    for config_name, config_results in results.items():
        stats = aggregate_results(config_results)

        if metric == "steps_to_badge":
            mean = stats.get("steps_to_badge_mean", {}).get(badge, float("inf"))
            std = stats.get("steps_to_badge_std", {}).get(badge, 0.0)
        elif metric == "final_badges":
            mean = stats.get("badges_mean", 0.0)
            std = stats.get("badges_std", 0.0)
        elif metric == "final_events":
            mean = stats.get("events_mean", 0.0)
            std = stats.get("events_std", 0.0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        rankings.append((config_name, mean, std))

    # Sort by mean value
    # For steps: ascending (lower is better)
    # For badges/events: descending (higher is better)
    if metric == "steps_to_badge":
        rankings.sort(key=lambda x: x[1])  # Ascending
    else:
        rankings.sort(key=lambda x: x[1], reverse=True)  # Descending

    return rankings


def compare_results(
    results_a: list[BenchmarkResult],
    results_b: list[BenchmarkResult],
    badge: int = 1,
) -> ComparisonResult:
    """
    Compare two configurations on steps to a specific badge.

    Args:
        results_a: Results from configuration A.
        results_b: Results from configuration B.
        badge: Badge number to compare (1-8).

    Returns:
        ComparisonResult with detailed comparison data.

    Example:
        >>> comparison = compare_results(baseline_results, high_lr_results)
        >>> if comparison.winner == "config_b":
        ...     print(f"high_lr is {comparison.improvement_pct:.1f}% faster")
    """
    stats_a = aggregate_results(results_a)
    stats_b = aggregate_results(results_b)

    config_a = stats_a.get("config_name", "config_a")
    config_b = stats_b.get("config_name", "config_b")

    # Get steps to badge
    value_a = stats_a.get("steps_to_badge_mean", {}).get(badge, float("inf"))
    value_b = stats_b.get("steps_to_badge_mean", {}).get(badge, float("inf"))

    # Calculate difference and improvement
    if value_a == float("inf") and value_b == float("inf"):
        difference = 0.0
        improvement_pct = 0.0
        winner = "tie"
    elif value_a == float("inf"):
        difference = float("-inf")
        improvement_pct = 100.0
        winner = config_b
    elif value_b == float("inf"):
        difference = float("inf")
        improvement_pct = -100.0
        winner = config_a
    else:
        difference = value_b - value_a
        # Improvement: positive if B is faster (lower steps)
        improvement_pct = ((value_a - value_b) / value_a) * 100

        # Determine winner (5% threshold to avoid noise)
        if improvement_pct > 5:
            winner = config_b
        elif improvement_pct < -5:
            winner = config_a
        else:
            winner = "tie"

    return ComparisonResult(
        config_a=config_a,
        config_b=config_b,
        metric=f"steps_to_badge_{badge}",
        value_a=value_a,
        value_b=value_b,
        difference=difference,
        improvement_pct=improvement_pct,
        winner=winner,
    )


def find_best_config(
    results: dict[str, list[BenchmarkResult]],
    badge: int = 1,
) -> tuple[str, dict[str, Any]]:
    """
    Find the configuration with best performance on steps to badge.

    Args:
        results: Dict mapping config names to lists of BenchmarkResult.
        badge: Badge number to optimize for (1-8).

    Returns:
        Tuple of (config_name, stats_dict) for the best configuration.

    Example:
        >>> best_name, best_stats = find_best_config(results, badge=1)
        >>> print(f"Best: {best_name} with {best_stats['steps_to_badge_mean'][1]:,.0f} steps")
    """
    rankings = rank_by_metric(results, badge=badge)
    if not rankings:
        return "none", {}

    best_name = rankings[0][0]
    best_results = results[best_name]
    best_stats = aggregate_results(best_results)

    return best_name, best_stats


def compute_improvement_matrix(
    results: dict[str, list[BenchmarkResult]],
    badge: int = 1,
) -> dict[str, dict[str, float]]:
    """
    Compute pairwise improvement percentages between all configurations.

    Returns a matrix where matrix[A][B] is the percentage improvement
    of B over A (positive means B is faster).

    Args:
        results: Dict mapping config names to lists of BenchmarkResult.
        badge: Badge number to compare (1-8).

    Returns:
        Nested dict: matrix[config_a][config_b] = improvement_pct of B over A.

    Example:
        >>> matrix = compute_improvement_matrix(results)
        >>> print(f"high_lr vs baseline: {matrix['baseline']['high_lr']:.1f}%")
    """
    config_names = list(results.keys())
    matrix: dict[str, dict[str, float]] = {name: {} for name in config_names}

    for config_a in config_names:
        for config_b in config_names:
            if config_a == config_b:
                matrix[config_a][config_b] = 0.0
            else:
                comparison = compare_results(
                    results[config_a],
                    results[config_b],
                    badge=badge,
                )
                matrix[config_a][config_b] = comparison.improvement_pct

    return matrix


def summarize_comparison(
    results: dict[str, list[BenchmarkResult]],
    badge: int = 1,
) -> dict[str, Any]:
    """
    Generate a comprehensive comparison summary.

    Args:
        results: Dict mapping config names to lists of BenchmarkResult.
        badge: Badge number for primary comparison.

    Returns:
        Dict with:
        - rankings: Ordered list from best to worst
        - best_config: Name of winning configuration
        - improvement_over_baseline: Improvement of best over first config
        - all_stats: Aggregated stats for each config

    Example:
        >>> summary = summarize_comparison(results)
        >>> print(f"Winner: {summary['best_config']}")
    """
    rankings = rank_by_metric(results, badge=badge)
    best_name, best_stats = find_best_config(results, badge=badge)

    # Get baseline (first config in results)
    baseline_name = next(iter(results.keys()), None)
    improvement = 0.0

    if baseline_name and baseline_name != best_name:
        comparison = compare_results(results[baseline_name], results[best_name], badge=badge)
        improvement = comparison.improvement_pct

    # Aggregate all stats
    all_stats = {
        config_name: aggregate_results(config_results)
        for config_name, config_results in results.items()
    }

    return {
        "rankings": rankings,
        "best_config": best_name,
        "best_stats": best_stats,
        "improvement_over_baseline": improvement,
        "all_stats": all_stats,
    }
