"""
Benchmark result reporters for KantoRL.

This module provides formatters for outputting benchmark results in various
formats: console (human-readable), JSON (machine-readable), and Markdown
(shareable reports).

Architecture Role:
    Reporters handle HOW results are presented. They take BenchmarkResult
    instances and format them for different use cases:
    - ConsoleReporter: Real-time progress and final summary
    - JSONReporter: Machine-readable output for automation
    - MarkdownReporter: Shareable reports with tables

Usage:
    >>> from kantorl.benchmarks.reporters import ConsoleReporter, JSONReporter
    >>>
    >>> # Console output
    >>> reporter = ConsoleReporter()
    >>> reporter.report_results(all_results)
    >>>
    >>> # JSON output
    >>> json_reporter = JSONReporter(output_path="results.json")
    >>> json_reporter.report_results(all_results)

Dependencies:
    - json: For JSON serialization
    - pathlib: For file path handling
    - datetime: For timestamps

References:
    - JSON specification: https://www.json.org/
    - GitHub Flavored Markdown: https://github.github.com/gfm/
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from kantorl.benchmarks.metrics import BenchmarkResult, aggregate_results

# =============================================================================
# BASE REPORTER
# =============================================================================


class BaseReporter:
    """
    Base class for benchmark reporters.

    Provides common interface and utility methods for all reporters.
    Subclasses must implement the report_results() method.

    Attributes:
        verbose: Verbosity level for progress output.
    """

    def __init__(self, verbose: int = 1):
        """Initialize reporter with verbosity level."""
        self.verbose = verbose

    def report_results(
        self,
        results: dict[str, list[BenchmarkResult]],
        **kwargs: Any,
    ) -> None:
        """
        Report benchmark results.

        Args:
            results: Dict mapping config names to lists of BenchmarkResult.
            **kwargs: Additional reporter-specific options.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement report_results()")


# =============================================================================
# CONSOLE REPORTER
# =============================================================================


class ConsoleReporter(BaseReporter):
    """
    Report benchmark results to the console with pretty formatting.

    Produces human-readable output with tables showing:
    - Per-config statistics (mean, std across seeds)
    - Steps to badge breakdown
    - Ranking by primary metric (steps to first badge)

    Example Output:
        ============================================================
        BENCHMARK RESULTS
        ============================================================

        Configuration: baseline (3 seeds)
        ────────────────────────────────────────────────────────────
        Badges:  2.3 ± 0.6
        Events:  145.0 ± 23.5
        Steps to Badge 1:  523,456 ± 45,678

        RANKINGS (by steps to badge 1)
        ────────────────────────────────────────────────────────────
        1. high_lr       450,234 steps (13.9% faster)
        2. baseline      523,456 steps (baseline)
        3. low_lr        678,901 steps (29.7% slower)
    """

    def __init__(self, verbose: int = 1, width: int = 60):
        """
        Initialize console reporter.

        Args:
            verbose: Verbosity level.
            width: Width of console output (for dividers).
        """
        super().__init__(verbose)
        self.width = width

    def _divider(self, char: str = "=") -> str:
        """Create a divider line."""
        return char * self.width

    def _header(self, title: str) -> str:
        """Create a header section."""
        return f"\n{self._divider()}\n{title}\n{self._divider()}\n"

    def report_results(
        self,
        results: dict[str, list[BenchmarkResult]],
        show_rankings: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Print benchmark results to console.

        Args:
            results: Dict mapping config names to lists of BenchmarkResult.
            show_rankings: Whether to show ranking table.
        """
        print(self._header("BENCHMARK RESULTS"))

        # Compute aggregated stats for each config
        aggregated: dict[str, dict[str, Any]] = {}
        for config_name, config_results in results.items():
            aggregated[config_name] = aggregate_results(config_results)

        # Print per-config statistics
        for config_name, stats in aggregated.items():
            n_runs = stats["n_runs"]
            print(f"\nConfiguration: {config_name} ({n_runs} seed{'s' if n_runs > 1 else ''})")
            print(self._divider("-"))

            # Badge and event stats
            print(f"  Badges:  {stats['badges_mean']:.1f} ± {stats['badges_std']:.1f}")
            print(f"  Events:  {stats['events_mean']:.0f} ± {stats['events_std']:.0f}")

            # Steps to each badge
            for badge, steps in stats.get("steps_to_badge_mean", {}).items():
                std = stats.get("steps_to_badge_std", {}).get(badge, 0)
                print(f"  Steps to Badge {badge}:  {steps:,.0f} ± {std:,.0f}")

        # Rankings
        if show_rankings and len(aggregated) > 1:
            self._print_rankings(aggregated)

    def _print_rankings(self, aggregated: dict[str, dict[str, Any]]) -> None:
        """Print ranking table by steps to first badge."""
        print(self._header("RANKINGS (by steps to badge 1)"))

        # Extract steps to badge 1 for each config
        rankings: list[tuple[str, float]] = []
        for config_name, stats in aggregated.items():
            steps_to_1 = stats.get("steps_to_badge_mean", {}).get(1, float("inf"))
            rankings.append((config_name, steps_to_1))

        # Sort by steps (lower is better)
        rankings.sort(key=lambda x: x[1])

        # Print rankings with comparison to baseline (first entry after sort)
        baseline_steps = rankings[0][1] if rankings else 0

        for i, (config_name, steps) in enumerate(rankings, 1):
            if steps == float("inf"):
                print(f"  {i}. {config_name:20s}  No badges reached")
            elif i == 1:
                print(f"  {i}. {config_name:20s}  {steps:,.0f} steps (winner)")
            else:
                pct_diff = ((steps - baseline_steps) / baseline_steps) * 100
                print(f"  {i}. {config_name:20s}  {steps:,.0f} steps (+{pct_diff:.1f}%)")


# =============================================================================
# JSON REPORTER
# =============================================================================


class JSONReporter(BaseReporter):
    """
    Report benchmark results as JSON for machine processing.

    Produces a JSON file with:
    - Metadata (timestamp, tier, etc.)
    - Raw results for each config and seed
    - Aggregated statistics

    Output Structure:
        {
          "metadata": {
            "timestamp": "2024-01-15T10:30:00",
            "tier": "bronze",
            "total_configs": 3
          },
          "results": {
            "baseline": [...],
            "high_lr": [...]
          },
          "aggregated": {
            "baseline": {"badges_mean": 2.3, ...},
            ...
          }
        }
    """

    def __init__(
        self,
        output_path: str | Path | None = None,
        verbose: int = 1,
        indent: int = 2,
    ):
        """
        Initialize JSON reporter.

        Args:
            output_path: Path to output JSON file. If None, prints to console.
            verbose: Verbosity level.
            indent: JSON indentation level.
        """
        super().__init__(verbose)
        self.output_path = Path(output_path) if output_path else None
        self.indent = indent

    def report_results(
        self,
        results: dict[str, list[BenchmarkResult]],
        tier: str = "unknown",
        **kwargs: Any,
    ) -> str:
        """
        Generate JSON report of benchmark results.

        Args:
            results: Dict mapping config names to lists of BenchmarkResult.
            tier: Tier name for metadata.

        Returns:
            JSON string of the report.
        """
        # Build report structure
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "tier": tier,
                "total_configs": len(results),
                "total_runs": sum(len(r) for r in results.values()),
            },
            "results": {
                config_name: [r.to_dict() for r in config_results]
                for config_name, config_results in results.items()
            },
            "aggregated": {
                config_name: aggregate_results(config_results)
                for config_name, config_results in results.items()
            },
        }

        # Serialize to JSON
        json_str = json.dumps(report, indent=self.indent, default=str)

        # Write to file or print
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w") as f:
                f.write(json_str)
            if self.verbose:
                print(f"JSON report saved to: {self.output_path}")
        else:
            print(json_str)

        return json_str


# =============================================================================
# MARKDOWN REPORTER
# =============================================================================


class MarkdownReporter(BaseReporter):
    """
    Report benchmark results as Markdown for sharing.

    Produces a Markdown document with:
    - Summary header
    - Results table with all configs
    - Per-config detail sections
    - Methodology notes

    Output Structure:
        # Benchmark Results

        ## Summary

        | Config | Badges | Events | Steps to Badge 1 |
        |--------|--------|--------|------------------|
        | baseline | 2.3 ± 0.6 | 145 ± 24 | 523,456 |
        ...

        ## Configuration Details
        ...
    """

    def __init__(
        self,
        output_path: str | Path | None = None,
        verbose: int = 1,
    ):
        """
        Initialize Markdown reporter.

        Args:
            output_path: Path to output .md file. If None, prints to console.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.output_path = Path(output_path) if output_path else None

    def report_results(
        self,
        results: dict[str, list[BenchmarkResult]],
        tier: str = "unknown",
        max_steps: int = 0,
        **kwargs: Any,
    ) -> str:
        """
        Generate Markdown report of benchmark results.

        Args:
            results: Dict mapping config names to lists of BenchmarkResult.
            tier: Tier name for header.
            max_steps: Max steps limit for methodology note.

        Returns:
            Markdown string of the report.
        """
        lines: list[str] = []

        # Header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append("# KantoRL Benchmark Results")
        lines.append(f"\n**Generated:** {timestamp}")
        lines.append(f"**Tier:** {tier.title()}")
        lines.append(f"**Max Steps:** {max_steps:,}")
        lines.append("")

        # Compute aggregated stats
        aggregated: dict[str, dict[str, Any]] = {}
        for config_name, config_results in results.items():
            aggregated[config_name] = aggregate_results(config_results)

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Config | Seeds | Badges | Events | Steps to Badge 1 |")
        lines.append("|--------|-------|--------|--------|------------------|")

        for config_name, stats in aggregated.items():
            n_runs = stats["n_runs"]
            badges = f"{stats['badges_mean']:.1f} ± {stats['badges_std']:.1f}"
            events = f"{stats['events_mean']:.0f} ± {stats['events_std']:.0f}"
            steps_to_1 = stats.get("steps_to_badge_mean", {}).get(1)
            if steps_to_1:
                std = stats.get("steps_to_badge_std", {}).get(1, 0)
                steps_str = f"{steps_to_1:,.0f} ± {std:,.0f}"
            else:
                steps_str = "N/A"
            lines.append(f"| {config_name} | {n_runs} | {badges} | {events} | {steps_str} |")

        lines.append("")

        # Rankings section
        lines.append("## Rankings")
        lines.append("")
        lines.append("Ranked by steps to badge 1 (lower is better):")
        lines.append("")

        rankings: list[tuple[str, float]] = []
        for config_name, stats in aggregated.items():
            steps_to_1 = stats.get("steps_to_badge_mean", {}).get(1, float("inf"))
            rankings.append((config_name, steps_to_1))
        rankings.sort(key=lambda x: x[1])

        baseline = rankings[0][1] if rankings else 0
        for i, (config_name, steps) in enumerate(rankings, 1):
            if steps == float("inf"):
                lines.append(f"{i}. **{config_name}**: No badges reached")
            elif i == 1:
                lines.append(f"{i}. **{config_name}**: {steps:,.0f} steps (winner)")
            else:
                pct = ((steps - baseline) / baseline) * 100
                lines.append(f"{i}. **{config_name}**: {steps:,.0f} steps (+{pct:.1f}%)")

        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by KantoRL Benchmark System*")

        # Join and output
        md_str = "\n".join(lines)

        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w") as f:
                f.write(md_str)
            if self.verbose:
                print(f"Markdown report saved to: {self.output_path}")
        else:
            print(md_str)

        return md_str


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def report_all(
    results: dict[str, list[BenchmarkResult]],
    output_dir: str | Path = "runs/benchmarks",
    tier: str = "unknown",
    max_steps: int = 0,
) -> None:
    """
    Generate all report formats (console, JSON, Markdown).

    Convenience function that creates all three report types in one call.

    Args:
        results: Dict mapping config names to lists of BenchmarkResult.
        output_dir: Directory for output files.
        tier: Tier name for reports.
        max_steps: Max steps for methodology notes.

    Example:
        >>> report_all(results, output_dir="runs/benchmarks", tier="bronze")
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Console report
    console = ConsoleReporter()
    console.report_results(results)

    # JSON report
    json_path = output_dir / f"benchmark_{timestamp}.json"
    json_reporter = JSONReporter(output_path=json_path)
    json_reporter.report_results(results, tier=tier)

    # Markdown report
    md_path = output_dir / f"benchmark_{timestamp}.md"
    md_reporter = MarkdownReporter(output_path=md_path)
    md_reporter.report_results(results, tier=tier, max_steps=max_steps)
