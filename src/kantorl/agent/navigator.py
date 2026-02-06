"""
Map graph navigator with A* pathfinding for KantoRL agent.

This module provides map-to-map navigation using a connectivity graph
loaded from map_graph.json. It calculates shortest paths between maps
and provides potential-based reward shaping to guide the RL policy
toward the current quest objective.

Architecture Role:
    The navigator receives a goal map ID from the planner/manager and
    provides two outputs:
    1. Reward shaping: A small bonus/penalty each step based on whether
       the agent moved closer to or farther from the goal.
    2. Navigation encoding: An 8-dim observation hint showing direction
       and distance to the goal.

    Planner → goal_map_id → Navigator → {reward_shaping, nav_encoding}

Reward Shaping Theory:
    Uses potential-based reward shaping (Ng et al., 1999):
        F = gamma * phi(s') - phi(s)
    where phi(s) = -distance_to_goal.

    This guarantees that the optimal policy is preserved regardless of
    the shaping function — it only speeds up learning, never distorts it.

Design Decisions:
    - Map-level graph: Edges connect map IDs, not individual tiles.
      This keeps the graph tiny (<50 nodes) and pathfinding instant.
    - Lazy loading: Graph loaded on first use, cached per-process.
    - Distance heuristic: Uses global map coordinates from map_data.json
      as A* heuristic (admissible for map-level distances).
    - HM-aware: Edges can require HMs; paths respect current abilities.

Dependencies:
    - heapq: For A* priority queue
    - json/pathlib: For loading map_graph.json and map_data.json
    - numpy: For navigation encoding
    - kantorl.global_map: For map coordinate lookups
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# =============================================================================
# DATA TYPES
# =============================================================================


@dataclass
class MapEdge:
    """
    Represents a connection between two maps in the world graph.

    Attributes:
        from_map: Source map ID.
        to_map: Destination map ID.
        from_x: Exit X coordinate on the source map.
        from_y: Exit Y coordinate on the source map.
        to_x: Entry X coordinate on the destination map.
        to_y: Entry Y coordinate on the destination map.
        requires_hm: HM name required ("cut", "surf") or None.
        one_way: True if this edge is one-directional only.
    """

    from_map: int
    to_map: int
    from_x: int = 0
    from_y: int = 0
    to_x: int = 0
    to_y: int = 0
    requires_hm: str | None = None
    one_way: bool = False


# =============================================================================
# MAP GRAPH DATA LOADING
# =============================================================================

_MAP_GRAPH: dict[str, Any] | None = None
_MAP_POSITIONS: dict[int, tuple[int, int]] | None = None


def _load_map_graph() -> dict[str, Any]:
    """Load map connectivity graph from map_graph.json."""
    global _MAP_GRAPH
    if _MAP_GRAPH is None:
        data_path = Path(__file__).parents[1] / "data" / "map_graph.json"
        if data_path.exists():
            with open(data_path) as f:
                _MAP_GRAPH = json.load(f)
        else:
            _MAP_GRAPH = {"edges": [], "pokemon_centers": {}, "gyms": {}}
    return _MAP_GRAPH


def _load_map_positions() -> dict[int, tuple[int, int]]:
    """
    Load map center positions from map_data.json for A* heuristic.

    Returns a dict mapping map_id → (global_x, global_y).
    """
    global _MAP_POSITIONS
    if _MAP_POSITIONS is None:
        data_path = Path(__file__).parents[1] / "data" / "map_data.json"
        _MAP_POSITIONS = {}
        if data_path.exists():
            with open(data_path) as f:
                raw = json.load(f)
            for map_key, info in raw.items():
                try:
                    _MAP_POSITIONS[int(map_key)] = (info["x"], info["y"])
                except (KeyError, ValueError):
                    pass
    return _MAP_POSITIONS


def _build_adjacency() -> dict[int, list[tuple[int, str | None]]]:
    """
    Build adjacency list from the map graph edges.

    Returns:
        Dict mapping map_id → list of (neighbor_map_id, requires_hm).
    """
    graph_data = _load_map_graph()
    adj: dict[int, list[tuple[int, str | None]]] = {}

    for edge_data in graph_data.get("edges", []):
        from_map = edge_data["from"]
        to_map = edge_data["to"]
        requires = edge_data.get("requires_hm")
        one_way = edge_data.get("one_way", False)

        # Add forward edge
        adj.setdefault(from_map, []).append((to_map, requires))

        # Add reverse edge unless one-way
        if not one_way:
            adj.setdefault(to_map, []).append((from_map, requires))

    return adj


# =============================================================================
# NAVIGATOR
# =============================================================================


class Navigator:
    """
    Map graph navigator with A* pathfinding and reward shaping.

    Maintains a goal map ID and computes shortest paths through the
    map connectivity graph. Provides potential-based reward shaping
    to guide RL exploration toward the goal.

    This class is pickle-safe for SubprocVecEnv compatibility.
    All data files are loaded lazily and cached at the module level.

    Attributes:
        goal_map_id: Target map ID to navigate toward (-1 = no goal).
        _path_cache: Cached A* path for current goal.
        _prev_potential: Previous step's potential value for shaping.
    """

    __slots__ = ("goal_map_id", "_path_cache", "_prev_potential", "_adj")

    def __init__(self) -> None:
        """Initialize navigator with no goal."""
        self.goal_map_id: int = -1
        self._path_cache: list[int] = []
        self._prev_potential: float = 0.0
        self._adj: dict[int, list[tuple[int, str | None]]] | None = None

    def reset(self) -> None:
        """Clear navigation state. Called on episode reset."""
        self.goal_map_id = -1
        self._path_cache = []
        self._prev_potential = 0.0

    def set_goal(self, target_map_id: int) -> None:
        """
        Set the navigation target.

        Invalidates the path cache if the goal changed.

        Args:
            target_map_id: Map ID to navigate toward. -1 for no goal.
        """
        if target_map_id != self.goal_map_id:
            self.goal_map_id = target_map_id
            self._path_cache = []  # Invalidate cache

    def get_path(self, current_map_id: int) -> list[int]:
        """
        Get A* shortest path from current map to goal.

        Uses cached path if still valid (current map is on the path).
        Recomputes if the agent deviated or goal changed.

        Args:
            current_map_id: Current map ID.

        Returns:
            List of map IDs forming the path (including current).
            Empty list if no path exists or no goal set.
        """
        if self.goal_map_id < 0:
            return []

        if current_map_id == self.goal_map_id:
            return [current_map_id]

        # Check if cached path is still valid
        if self._path_cache and current_map_id in self._path_cache:
            idx = self._path_cache.index(current_map_id)
            return self._path_cache[idx:]

        # Recompute path
        self._path_cache = self._astar(current_map_id, self.goal_map_id)
        return self._path_cache

    def get_reward_shaping(
        self,
        prev_map: int,
        prev_x: int,
        prev_y: int,
        curr_map: int,
        curr_x: int,
        curr_y: int,
        gamma: float = 0.995,
    ) -> float:
        """
        Calculate potential-based reward shaping.

        Uses F = gamma * phi(s') - phi(s) where phi(s) = -distance_to_goal.
        This preserves the optimal policy (Ng et al., 1999).

        Args:
            prev_map: Previous step's map ID.
            prev_x: Previous step's X coordinate.
            prev_y: Previous step's Y coordinate.
            curr_map: Current step's map ID.
            curr_x: Current step's X coordinate.
            curr_y: Current step's Y coordinate.
            gamma: Discount factor (should match PPO's gamma).

        Returns:
            Reward shaping bonus. Positive when moving toward goal,
            negative when moving away, zero when stationary.
        """
        if self.goal_map_id < 0:
            return 0.0

        # Calculate potentials
        curr_potential = self._potential(curr_map, curr_x, curr_y)
        prev_potential = self._potential(prev_map, prev_x, prev_y)

        return gamma * curr_potential - prev_potential

    def get_nav_encoding(self, current_map_id: int) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Encode navigation state as an 8-dim float32 observation.

        Layout:
            [0-3]: Direction to next waypoint (one-hot: N/S/E/W)
            [4]: Normalized distance estimate (0=at goal, 1=far)
            [5]: Maps remaining to goal / 10 (capped at 1.0)
            [6]: At goal flag (1.0 if current map == goal)
            [7]: Has path flag (1.0 if valid path exists)

        Args:
            current_map_id: Current map ID.

        Returns:
            Float32 array of shape (8,).
        """
        encoding = np.zeros(8, dtype=np.float32)

        if self.goal_map_id < 0:
            return encoding

        path = self.get_path(current_map_id)

        if not path:
            return encoding

        # At goal flag
        if current_map_id == self.goal_map_id:
            encoding[6] = 1.0
            encoding[7] = 1.0
            return encoding

        # Has path flag
        encoding[7] = 1.0

        # Maps remaining
        encoding[5] = min(len(path) / 10.0, 1.0)

        # Direction to next waypoint
        if len(path) >= 2:
            next_map = path[1]
            positions = _load_map_positions()
            curr_pos = positions.get(current_map_id)
            next_pos = positions.get(next_map)

            if curr_pos and next_pos:
                dx = next_pos[0] - curr_pos[0]
                dy = next_pos[1] - curr_pos[1]

                # One-hot direction: N=0, S=1, E=2, W=3
                if abs(dy) >= abs(dx):
                    if dy < 0:
                        encoding[0] = 1.0  # North (up = negative Y)
                    else:
                        encoding[1] = 1.0  # South
                else:
                    if dx > 0:
                        encoding[2] = 1.0  # East
                    else:
                        encoding[3] = 1.0  # West

        # Distance estimate
        positions = _load_map_positions()
        goal_pos = positions.get(self.goal_map_id)
        curr_pos = positions.get(current_map_id)
        if goal_pos and curr_pos:
            dist = abs(goal_pos[0] - curr_pos[0]) + abs(goal_pos[1] - curr_pos[1])
            # Normalize: 444 is roughly max Kanto diagonal
            encoding[4] = min(dist / 444.0, 1.0)

        return encoding

    def find_nearest_pokemon_center(self, current_map_id: int) -> int:
        """
        Find the map ID of the nearest Pokemon Center's city.

        Used by the manager when a heal decision is made.

        Args:
            current_map_id: Current map ID.

        Returns:
            Map ID of the city containing the nearest Pokemon Center.
            Returns -1 if no centers are known.
        """
        graph_data = _load_map_graph()
        centers = graph_data.get("pokemon_centers", {})
        positions = _load_map_positions()

        if not centers:
            return -1

        curr_pos = positions.get(current_map_id, (0, 0))
        best_dist = float("inf")
        best_city = -1

        for center_info in centers.values():
            city_map = center_info.get("city_map", -1)
            city_pos = positions.get(city_map, (0, 0))
            dist = abs(city_pos[0] - curr_pos[0]) + abs(city_pos[1] - curr_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_city = city_map

        return best_city

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _get_adj(self) -> dict[int, list[tuple[int, str | None]]]:
        """Get or build the adjacency list (lazy, cached on instance)."""
        if self._adj is None:
            self._adj = _build_adjacency()
        return self._adj

    def _astar(self, start: int, goal: int) -> list[int]:
        """
        A* pathfinding on the map graph.

        Uses Manhattan distance on global coordinates as heuristic.

        Args:
            start: Starting map ID.
            goal: Target map ID.

        Returns:
            List of map IDs from start to goal (inclusive).
            Empty list if no path exists.
        """
        adj = self._get_adj()
        positions = _load_map_positions()

        goal_pos = positions.get(goal, (0, 0))

        def heuristic(map_id: int) -> float:
            pos = positions.get(map_id, (0, 0))
            return abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])

        # A* with (f_score, counter, map_id)
        # Counter breaks ties for deterministic behavior
        counter = 0
        open_set: list[tuple[float, int, int]] = [(heuristic(start), counter, start)]
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {start: 0.0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor, _requires_hm in adj.get(current, []):
                # For v1, ignore HM requirements (agent may not have them)
                # Future: check if party has the required HM move
                tentative_g = g_score[current] + 1.0

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        # No path found
        return []

    def _potential(self, map_id: int, x: int, y: int) -> float:
        """
        Calculate the potential function phi(s) for reward shaping.

        phi(s) = -distance_to_goal, so moving closer increases potential.

        Uses A* path length (map hops) plus intra-map distance to the
        exit point leading toward the goal.

        Args:
            map_id: Current map ID.
            x: Current X coordinate within the map.
            y: Current Y coordinate within the map.

        Returns:
            Potential value (higher = closer to goal).
        """
        if self.goal_map_id < 0:
            return 0.0

        if map_id == self.goal_map_id:
            return 0.0  # At goal: zero potential

        path = self.get_path(map_id)
        if not path:
            # No path: use global distance as fallback
            positions = _load_map_positions()
            curr_pos = positions.get(map_id, (0, 0))
            goal_pos = positions.get(self.goal_map_id, (0, 0))
            dist = abs(curr_pos[0] - goal_pos[0]) + abs(curr_pos[1] - goal_pos[1])
            return -dist / 100.0  # Normalize to reasonable range

        # Map-hop distance (each hop ≈ large distance)
        map_hops = len(path) - 1
        return -float(map_hops)
