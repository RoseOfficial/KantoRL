"""
Resource management heuristics for KantoRL agent.

This module provides rule-based decisions about when to heal, when to
grind, and when to continue toward the quest objective. It does NOT
control actions directly — instead, it overrides the navigator's goal.

Architecture Role:
    The manager evaluates party health and level relative to the next
    gym leader, then advises the AgentWrapper:

    - "heal" → Navigator targets nearest Pokemon Center
    - "grind" → Navigator stays in current area (no goal override)
    - "continue" → Navigator targets planner's quest goal

    Manager → ManagerDecision → AgentWrapper → Navigator.set_goal()

Design Decisions:
    - Threshold-based: Simple HP% and level-gap checks. These are
      well-established heuristics for Pokemon Red speedruns.
    - No learning: Resource management in early Pokemon Red is a
      solved problem — heal below 40%, grind if underleveled.
    - Gym target levels: Based on enemy gym leader levels + buffer.
    - Pickle-safe: No state that can't be pickled.

Dependencies:
    - kantorl.memory: For reading party HP and levels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kantorl.agent.planner import QuestGoal

if TYPE_CHECKING:
    from pyboy import PyBoy  # type: ignore[import-untyped]

from kantorl import memory

# =============================================================================
# MANAGER DECISION
# =============================================================================


@dataclass
class ManagerDecision:
    """
    Resource management decision from the manager module.

    Attributes:
        action: Decision type: "heal", "grind", or "continue".
        reason: Human-readable explanation for logging.
        priority: Decision confidence (0.0-1.0). Higher = more urgent.
        target_map_id: Map to navigate to, or None for current area.
    """

    action: str
    reason: str
    priority: float
    target_map_id: int | None = None


# =============================================================================
# GYM TARGET LEVELS
# =============================================================================
# Target levels for each gym leader, based on their strongest Pokemon's
# level plus a small buffer. The agent should be at least this level
# before challenging the gym.
#
# Source: Pokemon Red gym leader teams (bulbapedia)

GYM_TARGET_LEVELS: dict[int, int] = {
    0: 12,   # Brock (highest: Onix Lv14, target Lv12)
    1: 21,   # Misty (highest: Starmie Lv21, target Lv21)
    2: 24,   # Surge (highest: Raichu Lv24, target Lv24)
    3: 29,   # Erika (highest: Vileplume Lv29, target Lv29)
    4: 43,   # Koga (highest: Weezing Lv43, target Lv43)
    5: 43,   # Sabrina (highest: Alakazam Lv43, target Lv43)
    6: 47,   # Blaine (highest: Arcanine Lv47, target Lv47)
    7: 50,   # Giovanni (highest: Rhydon Lv50, target Lv50)
}


# =============================================================================
# RESOURCE MANAGER
# =============================================================================

# Party HP fraction threshold — heal when below this
_HEAL_THRESHOLD = 0.4

# Level gap tolerance — grind when more than this many levels below target
_GRIND_LEVEL_GAP = 3


class ResourceManager:
    """
    Rule-based resource management for the agent.

    Evaluates party health and Pokemon levels to decide whether the
    agent should heal, grind, or continue toward the quest goal.

    This class is pickle-safe for SubprocVecEnv compatibility.
    """

    __slots__: list[str] = []

    def reset(self) -> None:
        """Reset per-episode state. Currently stateless."""

    def evaluate(
        self,
        pyboy: PyBoy,
        current_goal: QuestGoal,
        badges: int,
    ) -> ManagerDecision:
        """
        Evaluate current resources and recommend an action.

        Checks party HP and levels against thresholds, factoring
        in the current quest goal and badge count.

        Args:
            pyboy: PyBoy emulator instance.
            current_goal: Current quest goal from the planner.
            badges: Number of badges earned (determines gym target level).

        Returns:
            ManagerDecision with recommended action.
        """
        # Read party health
        current_hp, max_hp = memory.get_total_party_hp(pyboy)
        hp_fraction = current_hp / max(max_hp, 1)

        # Check if healing is needed
        if hp_fraction < _HEAL_THRESHOLD and max_hp > 0:
            return ManagerDecision(
                action="heal",
                reason=f"Party HP critical: {hp_fraction:.0%}",
                priority=1.0 - hp_fraction,  # More urgent at lower HP
                target_map_id=None,  # Navigator will find nearest Pokemon Center
            )

        # Check if grinding is needed
        levels = memory.get_party_levels(pyboy)
        if levels:
            max_level = max(levels)
            target_level = GYM_TARGET_LEVELS.get(badges, 12)

            if max_level < target_level - _GRIND_LEVEL_GAP:
                return ManagerDecision(
                    action="grind",
                    reason=(
                        f"Underleveled: Lv{max_level} vs target Lv{target_level}"
                    ),
                    priority=0.5,
                    target_map_id=None,  # Stay in current area
                )

        # Default: continue toward quest goal
        return ManagerDecision(
            action="continue",
            reason="Resources adequate",
            priority=0.0,
            target_map_id=current_goal.target_map_id,
        )
