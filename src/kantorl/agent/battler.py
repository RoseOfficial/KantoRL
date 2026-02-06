"""
Rule-based battle policy for KantoRL agent.

This module provides a heuristic battle system that uses type effectiveness,
STAB (Same-Type Attack Bonus), and move power to make tactical decisions.
This is NOT learned via RL — it encodes known-optimal battle strategies.

Architecture Role:
    The battler receives the current battle state (read from memory by
    the AgentWrapper) and returns a BattleAction (FIGHT_1..4, SWITCH, RUN).
    The controller then converts this into button sequences.

    Memory → BattleState → RuleBasedBattler → BattleAction → Controller

Strategy (priority order):
    1. Wild battle + party HP critical → RUN
    2. Current Pokemon has super-effective move with PP → use it
    3. Current Pokemon type-disadvantaged + can switch → SWITCH
    4. Use highest-power STAB move with PP remaining
    5. Use highest-power move with PP remaining
    6. Fallback: FIGHT_1 (game handles Struggle when all PP depleted)

Design Decisions:
    - Pure heuristic: No learning, no exploration. Battle strategy in
      Pokemon Red is a solved problem for most encounters.
    - Type chart from JSON: Loaded lazily, same caching pattern as global_map.
    - Observation encoding: Provides a 64-dim float vector for the RL policy
      even though the battler handles actions. This helps the RL value function
      estimate battle outcomes for better exploration decisions.

Dependencies:
    - numpy: For battle observation encoding
    - json/pathlib: For loading type chart and move data
    - kantorl.memory: For GAME_TYPE_TO_INDEX mapping
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kantorl.agent.controller import BattleAction
from kantorl.memory import GAME_TYPE_TO_INDEX

# =============================================================================
# DATA FILE LOADING (LAZY CACHED)
# =============================================================================

_TYPE_CHART: list[list[float]] | None = None
_MOVE_DATA: dict[str, dict[str, Any]] | None = None
_SPECIES_DATA: dict[str, dict[str, Any]] | None = None


def _load_type_chart() -> list[list[float]]:
    """Load 15x15 type effectiveness chart from type_chart.json."""
    global _TYPE_CHART
    if _TYPE_CHART is None:
        data_path = Path(__file__).parents[1] / "data" / "type_chart.json"
        if data_path.exists():
            with open(data_path) as f:
                data = json.load(f)
            _TYPE_CHART = data.get("chart", [])
        else:
            # Fallback: all neutral effectiveness
            _TYPE_CHART = [[1.0] * 15 for _ in range(15)]
    return _TYPE_CHART


def _load_move_data() -> dict[str, dict[str, Any]]:
    """Load move data from move_data.json."""
    global _MOVE_DATA
    if _MOVE_DATA is None:
        data_path = Path(__file__).parents[1] / "data" / "move_data.json"
        if data_path.exists():
            with open(data_path) as f:
                _MOVE_DATA = json.load(f)
        else:
            _MOVE_DATA = {}
    return _MOVE_DATA


def _load_species_data() -> dict[str, dict[str, Any]]:
    """Load species data from species_data.json."""
    global _SPECIES_DATA
    if _SPECIES_DATA is None:
        data_path = Path(__file__).parents[1] / "data" / "species_data.json"
        if data_path.exists():
            with open(data_path) as f:
                _SPECIES_DATA = json.load(f)
        else:
            _SPECIES_DATA = {}
    return _SPECIES_DATA


# =============================================================================
# BATTLE STATE DATACLASS
# =============================================================================


@dataclass
class BattleState:
    """
    Snapshot of the current battle state, read from memory.

    Contains normalized information about both the player's active
    Pokemon and the enemy Pokemon, suitable for decision-making.

    Attributes:
        player_species: Internal species ID of player's active Pokemon.
        player_hp: Current HP of player's active Pokemon.
        player_max_hp: Maximum HP of player's active Pokemon.
        player_level: Level of player's active Pokemon.
        player_types: (type_index_1, type_index_2) using chart indices.
        player_moves: List of 4 move IDs (0 = empty).
        player_pp: List of 4 PP values.
        player_status: Status condition byte.
        player_stat_mods: List of 6 stat stages (7 = neutral).
        enemy_species: Internal species ID of enemy Pokemon.
        enemy_hp: Current HP of enemy Pokemon.
        enemy_max_hp: Maximum HP of enemy Pokemon.
        enemy_level: Level of enemy Pokemon.
        enemy_types: (type_index_1, type_index_2) using chart indices.
        enemy_status: Status condition byte.
        enemy_stat_mods: List of 6 stat stages (7 = neutral).
        is_wild: True if this is a wild Pokemon battle.
        party_hp_fraction: Total party HP as fraction [0, 1].
    """

    player_species: int = 0
    player_hp: int = 0
    player_max_hp: int = 1
    player_level: int = 1
    player_types: tuple[int, int] = (0, 0)
    player_moves: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    player_pp: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    player_status: int = 0
    player_stat_mods: list[int] = field(default_factory=lambda: [7] * 6)
    enemy_species: int = 0
    enemy_hp: int = 0
    enemy_max_hp: int = 1
    enemy_level: int = 1
    enemy_types: tuple[int, int] = (0, 0)
    enemy_status: int = 0
    enemy_stat_mods: list[int] = field(default_factory=lambda: [7] * 6)
    is_wild: bool = True
    party_hp_fraction: float = 1.0

    @classmethod
    def from_memory(
        cls,
        player_data: dict[str, Any],
        enemy_data: dict[str, Any],
        player_mods: list[int],
        enemy_mods: list[int],
        battle_type: int,
        party_hp_fraction: float,
    ) -> BattleState:
        """
        Create BattleState from memory-read dictionaries.

        Converts game-internal type IDs to chart indices using
        GAME_TYPE_TO_INDEX mapping.

        Args:
            player_data: Dict from memory.get_battle_player_pokemon().
            enemy_data: Dict from memory.get_battle_enemy_pokemon().
            player_mods: List from memory.get_stat_modifiers(is_player=True).
            enemy_mods: List from memory.get_stat_modifiers(is_player=False).
            battle_type: 0=wild, 1=trainer from memory.get_battle_type().
            party_hp_fraction: Total party HP fraction [0, 1].
        """
        p_moves = player_data.get("moves", [0, 0, 0, 0])
        p_pp = player_data.get("pp", [0, 0, 0, 0])

        # Ensure lists are the right length
        if not isinstance(p_moves, list):
            p_moves = [0, 0, 0, 0]
        if not isinstance(p_pp, list):
            p_pp = [0, 0, 0, 0]

        return cls(
            player_species=int(player_data.get("species", 0)),
            player_hp=int(player_data.get("hp", 0)),
            player_max_hp=max(int(player_data.get("max_hp", 1)), 1),
            player_level=int(player_data.get("level", 1)),
            player_types=(
                GAME_TYPE_TO_INDEX.get(int(player_data.get("type1", 0)), 0),
                GAME_TYPE_TO_INDEX.get(int(player_data.get("type2", 0)), 0),
            ),
            player_moves=list(p_moves),
            player_pp=list(p_pp),
            player_status=int(player_data.get("status", 0)),
            player_stat_mods=player_mods,
            enemy_species=int(enemy_data.get("species", 0)),
            enemy_hp=int(enemy_data.get("hp", 0)),
            enemy_max_hp=max(int(enemy_data.get("max_hp", 1)), 1),
            enemy_level=int(enemy_data.get("level", 1)),
            enemy_types=(
                GAME_TYPE_TO_INDEX.get(int(enemy_data.get("type1", 0)), 0),
                GAME_TYPE_TO_INDEX.get(int(enemy_data.get("type2", 0)), 0),
            ),
            enemy_status=int(enemy_data.get("status", 0)),
            enemy_stat_mods=enemy_mods,
            is_wild=(battle_type == 0),
            party_hp_fraction=party_hp_fraction,
        )


# =============================================================================
# RULE-BASED BATTLER
# =============================================================================

# HP threshold below which we flee from wild battles
_CRITICAL_HP_THRESHOLD = 0.25


class RuleBasedBattler:
    """
    Heuristic battle policy using type chart and move data.

    Makes tactical decisions based on type effectiveness, STAB bonus,
    move power, and party health status. Does not use RL.

    This class is pickle-safe for SubprocVecEnv compatibility.
    All data files are loaded lazily and cached at the module level.
    """

    __slots__: list[str] = []

    def reset(self) -> None:
        """Reset per-episode battle state. Currently stateless."""

    def decide_action(self, state: BattleState) -> BattleAction:
        """
        Decide what to do in the current battle.

        Priority:
            1. Wild + critical HP → RUN
            2. Super-effective move with PP → use it
            3. Type-disadvantaged + can switch → SWITCH
            4. Best STAB move with PP → use it
            5. Best power move with PP → use it
            6. Fallback → FIGHT_1

        Args:
            state: Current battle snapshot.

        Returns:
            BattleAction to take.
        """
        # Rule 1: Flee from wild battles when HP is critical
        if state.is_wild and state.party_hp_fraction < _CRITICAL_HP_THRESHOLD:
            return BattleAction.RUN

        # Score each available move
        move_scores = self._score_moves(state)

        # Find best move that has PP
        best_slot = -1
        best_score = -1.0
        for slot in range(4):
            move_id = state.player_moves[slot]
            pp = state.player_pp[slot] if slot < len(state.player_pp) else 0
            if move_id == 0 or pp <= 0:
                continue
            if move_scores[slot] > best_score:
                best_score = move_scores[slot]
                best_slot = slot

        # If we found a move, use it
        if best_slot >= 0:
            return BattleAction(best_slot)  # FIGHT_1..FIGHT_4

        # Fallback: first slot (game handles Struggle)
        return BattleAction.FIGHT_1

    def _score_moves(self, state: BattleState) -> list[float]:
        """
        Score each move based on power, type effectiveness, and STAB.

        Scoring formula: base_power × effectiveness × stab_bonus
        - effectiveness: Type chart lookup (0.0, 0.5, 1.0, or 2.0)
        - stab_bonus: 1.5 if move type matches Pokemon type, else 1.0

        Args:
            state: Current battle snapshot.

        Returns:
            List of 4 scores (higher = better). 0.0 for empty/no-PP moves.
        """
        move_data = _load_move_data()
        scores = [0.0, 0.0, 0.0, 0.0]

        for slot in range(4):
            move_id = state.player_moves[slot]
            pp = state.player_pp[slot] if slot < len(state.player_pp) else 0
            if move_id == 0 or pp <= 0:
                continue

            move_info = move_data.get(str(move_id))
            if move_info is None:
                # Unknown move: use low default score
                scores[slot] = 10.0
                continue

            power = move_info.get("power", 0)
            if power == 0:
                # Status move: very low priority in heuristic battler
                scores[slot] = 1.0
                continue

            move_type = move_info.get("type", 0)

            # Type effectiveness against enemy
            effectiveness = self.get_type_effectiveness(
                move_type, state.enemy_types[0], state.enemy_types[1]
            )

            # STAB bonus (Same-Type Attack Bonus)
            stab = 1.5 if (
                move_type == state.player_types[0]
                or move_type == state.player_types[1]
            ) else 1.0

            scores[slot] = power * effectiveness * stab

        return scores

    @staticmethod
    def get_type_effectiveness(
        atk_type: int,
        def_type1: int,
        def_type2: int,
    ) -> float:
        """
        Look up type effectiveness from the chart.

        For dual-typed defenders, multiplies effectiveness against
        both types (e.g., Water vs Fire/Flying = 2.0 × 1.0 = 2.0).

        Args:
            atk_type: Attacking move's type index (0-14).
            def_type1: Defender's primary type index (0-14).
            def_type2: Defender's secondary type index (0-14).

        Returns:
            Combined effectiveness multiplier (0.0, 0.25, 0.5, 1.0, 2.0, 4.0).
        """
        chart = _load_type_chart()
        if not chart or atk_type >= len(chart):
            return 1.0

        row = chart[atk_type]

        eff1 = row[def_type1] if def_type1 < len(row) else 1.0
        eff2 = row[def_type2] if def_type2 < len(row) else 1.0

        # If both types are the same, don't double-apply
        if def_type1 == def_type2:
            return eff1

        return eff1 * eff2

    @staticmethod
    def get_battle_observation(state: BattleState) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Encode battle state as a float32 observation vector.

        Provides a 64-dimensional encoding of the battle for the RL
        policy's observation space. Element 0 is the in_battle flag.

        Layout (64 dims):
            [0]:    in_battle flag (1.0 during battle)
            [1-2]:  player HP fraction, enemy HP fraction
            [3-4]:  player level / 100, enemy level / 100
            [5-19]: player type one-hot (15 dims)
            [20-34]: enemy type one-hot (15 dims)
            [35-38]: move type effectiveness scores (4 moves)
            [39-42]: move power / 200 (4 moves, normalized)
            [43]:   is_wild flag
            [44]:   party HP fraction
            [45-50]: player stat mod deltas from neutral (6 dims)
            [51-56]: enemy stat mod deltas from neutral (6 dims)
            [57]:   player has status condition
            [58]:   enemy has status condition
            [59-63]: reserved for future use

        Args:
            state: Current battle snapshot.

        Returns:
            Float32 array of shape (64,).
        """
        obs = np.zeros(64, dtype=np.float32)

        # Battle flag
        obs[0] = 1.0

        # HP fractions
        obs[1] = state.player_hp / max(state.player_max_hp, 1)
        obs[2] = state.enemy_hp / max(state.enemy_max_hp, 1)

        # Levels (normalized)
        obs[3] = state.player_level / 100.0
        obs[4] = state.enemy_level / 100.0

        # Player type one-hot (15 dims starting at index 5)
        if 0 <= state.player_types[0] < 15:
            obs[5 + state.player_types[0]] = 1.0
        if 0 <= state.player_types[1] < 15:
            obs[5 + state.player_types[1]] = 1.0

        # Enemy type one-hot (15 dims starting at index 20)
        if 0 <= state.enemy_types[0] < 15:
            obs[20 + state.enemy_types[0]] = 1.0
        if 0 <= state.enemy_types[1] < 15:
            obs[20 + state.enemy_types[1]] = 1.0

        # Move effectiveness scores (4 dims starting at index 35)
        move_data = _load_move_data()
        for slot in range(4):
            move_id = state.player_moves[slot]
            if move_id == 0:
                continue
            move_info = move_data.get(str(move_id))
            if move_info is None:
                continue
            move_type = move_info.get("type", 0)
            eff = RuleBasedBattler.get_type_effectiveness(
                move_type, state.enemy_types[0], state.enemy_types[1]
            )
            obs[35 + slot] = eff

            # Move power normalized
            obs[39 + slot] = move_info.get("power", 0) / 200.0

        # Battle metadata
        obs[43] = 1.0 if state.is_wild else 0.0
        obs[44] = state.party_hp_fraction

        # Stat mod deltas from neutral (7 = neutral)
        for i in range(6):
            if i < len(state.player_stat_mods):
                obs[45 + i] = (state.player_stat_mods[i] - 7) / 6.0
        for i in range(6):
            if i < len(state.enemy_stat_mods):
                obs[51 + i] = (state.enemy_stat_mods[i] - 7) / 6.0

        # Status conditions
        obs[57] = 1.0 if state.player_status != 0 else 0.0
        obs[58] = 1.0 if state.enemy_status != 0 else 0.0

        return obs
