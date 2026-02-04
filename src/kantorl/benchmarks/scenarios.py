"""
Milestone tier definitions for KantoRL benchmarks.

This module defines the benchmark milestone system - a tiered progression of
goals that measure how far an agent has progressed through Pokemon Red. Each
tier represents a meaningful checkpoint in the game:

- **Bronze**: First badge (Brock) + basic events - fast iteration during development
- **Silver**: Four badges (half gyms) + mid-game events - validation checkpoint
- **Gold**: All eight badges - full gym completion
- **Champion**: Elite Four defeated - complete game victory

Architecture Role:
    Scenarios define WHAT we measure. They provide:
    1. Clear, reproducible goals for benchmarking
    2. Tiered difficulty for different use cases
    3. Threshold definitions for automated checking

    The runner (runner.py) uses these scenarios to track progress during training,
    and the comparator (comparator.py) uses them to rank configurations.

Primary Metric:
    Steps to reach milestone (lower is better). This measures learning efficiency.
    Secondary metrics include wall-clock time and events triggered.

Usage:
    >>> from kantorl.benchmarks.scenarios import MilestoneTier, check_milestone_reached
    >>>
    >>> # Check if Bronze milestone is reached
    >>> thresholds = get_tier_thresholds(MilestoneTier.BRONZE)
    >>> is_reached = check_milestone_reached(badges=1, events=60, tier=MilestoneTier.BRONZE)
    >>> print(f"Bronze reached: {is_reached}")

Dependencies:
    - enum: For MilestoneTier enum definition

References:
    - Pokemon Red gym order: Brock → Misty → Lt. Surge → Erika →
      Koga → Sabrina → Blaine → Giovanni
    - Elite Four: Lorelei → Bruno → Agatha → Lance → Champion (rival)
"""

from dataclasses import dataclass
from enum import Enum

# =============================================================================
# MILESTONE TIER ENUM
# =============================================================================


class MilestoneTier(Enum):
    """
    Benchmark milestone tiers representing game progress checkpoints.

    Each tier defines a specific goal in Pokemon Red progression. Tiers are
    ordered by difficulty, from Bronze (easiest) to Champion (hardest).

    Attributes:
        BRONZE: First badge collected (Brock's Boulder Badge).
                Useful for fast iteration during development.
                Typical training: 500K - 2M steps.

        SILVER: Four badges collected (half of all gyms).
                Mid-game validation checkpoint.
                Typical training: 5M - 20M steps.

        GOLD: All eight gym badges collected.
              Full gym completion without Elite Four.
              Typical training: 20M - 100M steps.

        CHAMPION: Elite Four defeated, game completed.
                  The ultimate benchmark goal.
                  Typical training: 100M+ steps.

    Usage:
        >>> tier = MilestoneTier.BRONZE
        >>> print(f"Running benchmark for tier: {tier.value}")
        Running benchmark for tier: bronze

    Notes:
        - Tier names match Pokemon medal aesthetics (Bronze/Silver/Gold)
        - Champion is special - represents complete game victory
        - Lower tiers are subsets of higher tiers (Gold implies Silver implies Bronze)
    """

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    CHAMPION = "champion"


# =============================================================================
# TIER THRESHOLDS
# =============================================================================


@dataclass(frozen=True)
class TierThresholds:
    """
    Threshold values that define when a milestone tier is reached.

    This dataclass holds the minimum requirements for each tier. All conditions
    must be met for the tier to be considered "reached".

    Attributes:
        badges: Minimum number of gym badges required.
        events: Minimum number of event flags that must be triggered.
        description: Human-readable description of this tier.

    Example:
        >>> bronze = TierThresholds(badges=1, events=50, description="First badge")
        >>> print(f"Need {bronze.badges} badge(s) and {bronze.events}+ events")

    Notes:
        - frozen=True makes instances immutable and hashable
        - Events provide fine-grained progress beyond just badges
        - Event thresholds are estimates based on typical game progression
    """

    badges: int
    events: int
    description: str


# Tier threshold definitions
# These values are calibrated based on typical Pokemon Red progression
_TIER_THRESHOLDS: dict[MilestoneTier, TierThresholds] = {
    MilestoneTier.BRONZE: TierThresholds(
        badges=1,
        events=50,
        description="First badge (Brock) - Fast iteration during development",
    ),
    MilestoneTier.SILVER: TierThresholds(
        badges=4,
        events=200,
        description="Four badges (mid-game) - Mid-game validation",
    ),
    MilestoneTier.GOLD: TierThresholds(
        badges=8,
        events=500,
        description="All badges - Full gym completion",
    ),
    MilestoneTier.CHAMPION: TierThresholds(
        badges=8,
        events=700,
        description="Elite Four defeated - Complete game victory",
    ),
}


# =============================================================================
# THRESHOLD LOOKUP FUNCTIONS
# =============================================================================


def get_tier_thresholds(tier: MilestoneTier) -> TierThresholds:
    """
    Get the threshold values for a milestone tier.

    Args:
        tier: The milestone tier to get thresholds for.

    Returns:
        TierThresholds dataclass with badges, events, and description.

    Example:
        >>> thresholds = get_tier_thresholds(MilestoneTier.BRONZE)
        >>> print(f"Bronze requires {thresholds.badges} badge(s)")
        Bronze requires 1 badge(s)

    Raises:
        KeyError: If tier is not a valid MilestoneTier (should never happen).
    """
    return _TIER_THRESHOLDS[tier]


def check_milestone_reached(
    badges: int,
    events: int,
    tier: MilestoneTier,
) -> bool:
    """
    Check if the current game state meets the milestone tier requirements.

    Both badge count AND event count must meet or exceed the tier's thresholds
    for the milestone to be considered reached.

    Args:
        badges: Current number of gym badges (0-8).
        events: Current number of event flags triggered.
        tier: The milestone tier to check against.

    Returns:
        True if the tier's requirements are met, False otherwise.

    Example:
        >>> # Just got first badge with 60 events
        >>> check_milestone_reached(badges=1, events=60, tier=MilestoneTier.BRONZE)
        True
        >>> # Has badge but not enough events
        >>> check_milestone_reached(badges=1, events=30, tier=MilestoneTier.BRONZE)
        False

    Notes:
        - Both conditions must be met (AND logic)
        - Events help distinguish genuine progress from exploits
        - Champion tier uses events to verify Elite Four completion
    """
    thresholds = _TIER_THRESHOLDS[tier]
    return badges >= thresholds.badges and events >= thresholds.events


def get_badge_for_tier(tier: MilestoneTier) -> int:
    """
    Get the badge requirement for a tier.

    Convenience function to quickly check how many badges a tier requires.

    Args:
        tier: The milestone tier to check.

    Returns:
        Number of badges required for this tier.

    Example:
        >>> get_badge_for_tier(MilestoneTier.SILVER)
        4
    """
    return _TIER_THRESHOLDS[tier].badges


def get_all_tiers() -> list[MilestoneTier]:
    """
    Get all milestone tiers in order of difficulty.

    Returns:
        List of MilestoneTier values from easiest (Bronze) to hardest (Champion).

    Example:
        >>> for tier in get_all_tiers():
        ...     print(tier.value)
        bronze
        silver
        gold
        champion
    """
    return [
        MilestoneTier.BRONZE,
        MilestoneTier.SILVER,
        MilestoneTier.GOLD,
        MilestoneTier.CHAMPION,
    ]


def get_highest_reached_tier(badges: int, events: int) -> MilestoneTier | None:
    """
    Determine the highest milestone tier reached given current progress.

    Checks tiers from highest to lowest and returns the first one that is met.

    Args:
        badges: Current number of gym badges (0-8).
        events: Current number of event flags triggered.

    Returns:
        The highest MilestoneTier reached, or None if no tier is met.

    Example:
        >>> # With 4 badges and 250 events
        >>> tier = get_highest_reached_tier(badges=4, events=250)
        >>> print(tier.value if tier else "None")
        silver
    """
    # Check from highest to lowest tier
    for tier in reversed(get_all_tiers()):
        if check_milestone_reached(badges, events, tier):
            return tier
    return None
