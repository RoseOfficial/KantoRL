"""
Strategic quest planner for KantoRL agent.

This module implements a finite state machine (FSM) encoding Pokemon Red's
progression order. It is NOT learned via RL — it is hand-authored game
knowledge that tells the agent what to do next.

Architecture Role:
    The planner sits at the top of the agent hierarchy. It examines the
    current game state (badges, events, map) and determines the current
    high-level goal: "go to Pewter City", "defeat Brock", "deliver Oak's
    Parcel", etc. This goal is passed to the navigator for pathfinding
    and encoded into the observation space for the RL policy.

Design Decisions:
    - Data-driven FSM: State machine is loaded from quest_fsm.json,
      making it easy to extend without code changes.
    - Condition-based transitions: Each state checks game conditions
      (event counts, visited maps, badge counts) to advance.
    - Goal encoding: 16-dimensional float32 vector combining one-hot
      category, target map, and progress fraction.
    - Lazy loading: JSON file loaded on first use, cached per-process.

Dependencies:
    - numpy: For goal encoding arrays
    - json/pathlib: For loading quest_fsm.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# =============================================================================
# QUEST GOAL DATACLASS
# =============================================================================


@dataclass
class QuestGoal:
    """
    Represents the current strategic objective.

    Attributes:
        name: Machine-readable state name (e.g., "get_pokedex").
        target_map_id: Map ID to navigate toward (-1 if no specific target).
        description: Human-readable description for logging.
        required_events: Minimum event count for this goal's transitions.
        required_badges: Minimum badges needed before this goal activates.
    """

    name: str
    target_map_id: int
    description: str
    required_events: int = 0
    required_badges: int = 0


# =============================================================================
# QUEST FSM DATA LOADING
# =============================================================================

_QUEST_FSM_DATA: dict[str, Any] | None = None


def _load_quest_fsm() -> dict[str, Any]:
    """
    Load quest FSM data from the JSON file.

    Implements lazy loading with module-level caching — loaded once per
    process (important for SubprocVecEnv which forks per env).

    Returns:
        Dictionary with FSM states and transitions.
        Returns minimal fallback FSM if file not found.
    """
    global _QUEST_FSM_DATA
    if _QUEST_FSM_DATA is None:
        data_path = Path(__file__).parents[1] / "data" / "quest_fsm.json"
        if data_path.exists():
            with open(data_path) as f:
                _QUEST_FSM_DATA = json.load(f)
        else:
            # Fallback: single-state FSM that just explores
            _QUEST_FSM_DATA = {
                "initial_state": "explore",
                "states": {
                    "explore": {
                        "description": "Explore the world",
                        "target_map_id": -1,
                        "transitions": [],
                    }
                },
            }
    return _QUEST_FSM_DATA


# =============================================================================
# QUEST PLANNER
# =============================================================================


class QuestPlanner:
    """
    Finite state machine tracking Pokemon Red quest progression.

    The planner examines badges, event flags, and current map to determine
    which quest state the game is in, then provides a goal for the navigator.

    The FSM is loaded from quest_fsm.json and defines states like:
    - get_pokedex → route_1_north → viridian_to_pewter → beat_brock → ...

    Each state has transition conditions (min events, visited maps, badges)
    that advance the FSM when satisfied.

    This class is pickle-safe for SubprocVecEnv compatibility.

    Attributes:
        current_state: Name of the current FSM state.
        visited_maps: Set of map IDs visited this episode.
    """

    __slots__ = ("current_state", "visited_maps", "_fsm_data")

    def __init__(self) -> None:
        """Initialize planner with FSM data."""
        self._fsm_data = _load_quest_fsm()
        self.current_state: str = self._fsm_data["initial_state"]
        self.visited_maps: set[int] = set()

    def reset(self) -> None:
        """Reset FSM to initial state. Called on episode reset."""
        self.current_state = self._fsm_data["initial_state"]
        self.visited_maps = set()

    def update(
        self,
        badges: int,
        event_count: int,
        map_id: int,
    ) -> None:
        """
        Advance the FSM based on current game state.

        Checks all transition conditions for the current state and
        moves to the next state if any are satisfied.

        Args:
            badges: Number of badges earned (0-8).
            event_count: Total number of event flags set.
            map_id: Current map ID.
        """
        self.visited_maps.add(map_id)

        states = self._fsm_data.get("states", {})
        state_data = states.get(self.current_state)
        if state_data is None:
            return

        # Check each transition condition
        for transition in state_data.get("transitions", []):
            condition = transition.get("condition", {})
            satisfied = True

            # Check minimum event count
            if "min_events" in condition:
                if event_count < condition["min_events"]:
                    satisfied = False

            # Check minimum badges
            if "min_badges" in condition:
                if badges < condition["min_badges"]:
                    satisfied = False

            # Check if a specific map has been visited
            if "visited_map" in condition:
                if condition["visited_map"] not in self.visited_maps:
                    satisfied = False

            if satisfied:
                next_state = transition.get("next_state")
                if next_state and next_state in states:
                    self.current_state = next_state
                # Only take the first matching transition
                return

    def get_current_goal(self) -> QuestGoal:
        """
        Get the current quest goal from the FSM state.

        Returns:
            QuestGoal with the current objective's details.
        """
        states = self._fsm_data.get("states", {})
        state_data = states.get(self.current_state, {})

        return QuestGoal(
            name=self.current_state,
            target_map_id=state_data.get("target_map_id", -1),
            description=state_data.get("description", "Explore"),
            required_events=state_data.get("required_events", 0),
            required_badges=state_data.get("required_badges", 0),
        )

    def get_goal_encoding(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Encode the current goal as a float32 vector for observations.

        The encoding combines:
        - One-hot goal category (8 dims): which phase of the game
        - Target map ID (4 dims): normalized map coordinates
        - Progress fraction (4 dims): how far through the FSM

        Returns:
            Float32 array of shape (16,).
        """
        encoding = np.zeros(16, dtype=np.float32)

        states = self._fsm_data.get("states", {})
        state_names = list(states.keys())
        total_states = max(len(state_names), 1)

        # Find current state index
        try:
            state_idx = state_names.index(self.current_state)
        except ValueError:
            state_idx = 0

        # One-hot for goal category (first 8 dims)
        # Map state index to 8 categories (multiple states per category)
        category = min(state_idx * 8 // total_states, 7)
        encoding[category] = 1.0

        # Target map encoding (dims 8-11)
        goal = self.get_current_goal()
        if goal.target_map_id >= 0:
            # Normalize map ID to [0, 1] range (248 maps in Pokemon Red)
            encoding[8] = goal.target_map_id / 248.0
            encoding[9] = 1.0  # Has target flag
        # dims 10-11 reserved for future use

        # Progress fraction (dims 12-15)
        progress = state_idx / total_states
        encoding[12] = progress
        encoding[13] = goal.required_badges / 8.0
        # dims 14-15 reserved

        return encoding
