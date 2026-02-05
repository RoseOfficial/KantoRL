"""
HM field move automation for KantoRL curriculum learning.

This module automates the teaching and usage of HM field moves (Cut, Surf,
Strength) during curriculum-based training. Without this automation, the RL
agent hits hard walls at points requiring field moves, because the menu
navigation to use them is a ~20 button press sequence that random exploration
will never discover.

Architecture Role:
    HMAutomation is called by CurriculumWrapper.step() on every step. It:
    1. Periodically checks the bag for HM items and auto-teaches them
    2. Detects when the player faces obstacles requiring field moves
    3. Applies field move effects via direct memory writes

    CurriculumWrapper.step() → HMAutomation.step() → memory writes

Why Direct Memory Writes:
    Teaching HM moves via the in-game menu requires navigating:
    Start → Bag → Select HM → Use → Select Pokemon → Confirm
    This is ~20 button presses with specific timing, which is infeasible
    for RL exploration. Direct memory writes are:
    - Deterministic and instant (no menu navigation needed)
    - Safe (game reads move slots normally, doesn't validate how they were set)
    - Reliable (no timing or input registration issues)

Safety Guarantees:
    - Only writes to move slot 3 (4th slot, index 3) to minimize disruption
    - Only teaches HMs the Pokemon is actually compatible with (per Gen 1 data)
    - Only modifies 2 bytes per teach (move ID + PP value)
    - Field move effects only trigger when the correct tile is detected

Usage:
    >>> hm = HMAutomation(pyboy)
    >>> # Called every step by CurriculumWrapper
    >>> hm_info = hm.step(step_count=1000)
    >>> print(hm_info)  # {"hm_taught": ["cut"], "hm_used": []}

Dependencies:
    - kantorl.memory: Memory address constants and reading functions
    - kantorl.data.hm_compat: HM compatibility tables per species
"""

from typing import TYPE_CHECKING

from kantorl import memory
from kantorl.data.hm_compat import CUT_COMPATIBLE, STRENGTH_COMPATIBLE, SURF_COMPATIBLE

if TYPE_CHECKING:
    from pyboy import PyBoy


# =============================================================================
# HM AUTOMATION CONFIGURATION
# =============================================================================

# How often to check for new HM items in the bag (every N steps)
# Checking every step would waste cycles since HMs are rare pickups
# 500 steps ≈ 8 seconds of gameplay, fast enough to catch HM pickups
TEACH_CHECK_INTERVAL = 500


# =============================================================================
# HM MOVE DEFINITIONS
# =============================================================================
# Each HM definition links the bag item ID, the move ID to teach,
# the PP value for the move, and the compatible species set.

# Mapping: HM item ID → (move_id, pp, compatible_species)
HM_DEFINITIONS: dict[int, tuple[int, int, frozenset[int]]] = {
    memory.HM_CUT_ITEM: (memory.MOVE_CUT, memory.PP_CUT, CUT_COMPATIBLE),
    memory.HM_SURF_ITEM: (memory.MOVE_SURF, memory.PP_SURF, SURF_COMPATIBLE),
    memory.HM_STRENGTH_ITEM: (
        memory.MOVE_STRENGTH,
        memory.PP_STRENGTH,
        STRENGTH_COMPATIBLE,
    ),
}

# Human-readable names for logging
HM_NAMES: dict[int, str] = {
    memory.MOVE_CUT: "cut",
    memory.MOVE_SURF: "surf",
    memory.MOVE_STRENGTH: "strength",
}


# =============================================================================
# HM AUTOMATION CLASS
# =============================================================================


class HMAutomation:
    """
    Automates teaching and using HM field moves during curriculum training.

    This class handles two responsibilities:
    1. Auto-teach: When an HM item appears in the bag and no party Pokemon
       knows the corresponding move, teach it to the first compatible Pokemon
       by writing the move ID directly to memory.
    2. Auto-use: When the player faces a tile that requires a field move
       (cuttable tree, water, boulder), apply the effect directly.

    Attributes:
        pyboy: PyBoy emulator instance for memory access.
        taught_moves: Set of move IDs that have been auto-taught this episode.
            Used for logging and to avoid re-checking taught moves.

    Example:
        >>> hm = HMAutomation(pyboy)
        >>> info = hm.step(step_count=1000)
        >>> if info["hm_taught"]:
        ...     print(f"Auto-taught: {info['hm_taught']}")

    Notes:
        - Only teaches to move slot 3 (4th slot) to minimize disruption
        - Teaching writes exactly 2 bytes: move ID and PP value
        - Field move effects are applied via memory writes, not button input
        - Reset should be called at the start of each episode
    """

    def __init__(self, pyboy: "PyBoy") -> None:
        """
        Initialize HM automation.

        Args:
            pyboy: PyBoy emulator instance. Must have Pokemon Red loaded.
        """
        self.pyboy = pyboy

        # Track which moves have been auto-taught this episode
        # Prevents redundant teaching and provides logging info
        self.taught_moves: set[int] = set()

    def reset(self) -> None:
        """
        Reset automation state for a new episode.

        Clears the taught moves tracking. Called by CurriculumWrapper.reset().
        """
        self.taught_moves.clear()

    def step(self, step_count: int) -> dict[str, list[str]]:
        """
        Run one step of HM automation.

        Called every step by CurriculumWrapper. Checks for HM teaching
        opportunities periodically and field move usage every step.

        Args:
            step_count: Current step count in the episode. Used to determine
                when to check for HM items (every TEACH_CHECK_INTERVAL steps).

        Returns:
            Dictionary with automation info for logging:
            - "hm_taught": List of move names taught this step (e.g., ["cut"])
            - "hm_used": List of field moves used this step (e.g., ["surf"])

        Notes:
            - Teaching is checked every TEACH_CHECK_INTERVAL steps
            - Field move usage is checked every step (cheap memory reads)
            - Returns empty lists when no actions were taken
        """
        info: dict[str, list[str]] = {"hm_taught": [], "hm_used": []}

        # Periodically check for HM items to teach
        if step_count % TEACH_CHECK_INTERVAL == 0:
            taught = self._try_teach_hms()
            info["hm_taught"] = taught

        # Check for field move usage every step
        used = self._try_use_field_moves()
        info["hm_used"] = used

        return info

    def _try_teach_hms(self) -> list[str]:
        """
        Check bag for HM items and teach them to compatible party Pokemon.

        Scans the bag for HM01 (Cut), HM03 (Surf), HM04 (Strength).
        For each HM found, checks if any party Pokemon already knows the
        move. If not, finds the first compatible Pokemon and writes the
        move to their 4th move slot.

        Returns:
            List of move names that were taught (e.g., ["cut", "surf"]).

        Notes:
            - Only teaches to slot 3 (4th move slot) to avoid overwriting
              important moves in slots 0-2
            - Writes move ID and PP directly to memory
            - Only teaches if the Pokemon's species is in the compatibility set
        """
        taught: list[str] = []

        for hm_item_id, (move_id, pp, compatible) in HM_DEFINITIONS.items():
            # Skip if already taught this episode
            if move_id in self.taught_moves:
                continue

            # Check if HM item is in the bag
            if not memory.has_item(self.pyboy, hm_item_id):
                continue

            # Check if any party Pokemon already knows this move
            if memory.party_has_move(self.pyboy, move_id):
                self.taught_moves.add(move_id)
                continue

            # Find first compatible Pokemon in the party
            party_count = memory.get_party_count(self.pyboy)
            for slot in range(party_count):
                species_id = memory.read_byte(
                    self.pyboy, memory.ADDR_PARTY_SPECIES + slot
                )
                if species_id in compatible:
                    # Write move ID to the 4th move slot (index 3)
                    move_addr = memory.ADDR_PARTY_MOVES + slot * memory.PARTY_MON_STRIDE + 3
                    pp_addr = memory.ADDR_PARTY_PP + slot * memory.PARTY_MON_STRIDE + 3

                    self.pyboy.memory[move_addr] = move_id
                    self.pyboy.memory[pp_addr] = pp

                    self.taught_moves.add(move_id)
                    taught.append(HM_NAMES.get(move_id, f"move_{move_id}"))
                    break

        return taught

    def _try_use_field_moves(self) -> list[str]:
        """
        Check if a field move should be automatically used.

        Examines the tile the player is facing and the current movement
        state to determine if Cut, Surf, or Strength should be applied.

        Returns:
            List of field move names used this step (e.g., ["cut"]).

        Notes:
            - Cut: Removes cuttable tree tile by writing a walkable tile ID
            - Surf: Sets walk/bike/surf state to 2 (surfing) when facing water
            - Strength: Currently not implemented (boulder pushing is complex)
        """
        used: list[str] = []

        # Don't use field moves during battle
        if memory.is_in_battle(self.pyboy):
            return used

        tile = memory.get_tile_in_front(self.pyboy)

        # -----------------------------------------------------------------
        # Auto-Cut: Remove cuttable trees
        # -----------------------------------------------------------------
        if tile in memory.CUTTABLE_TREE_TILES:
            if memory.party_has_move(self.pyboy, memory.MOVE_CUT):
                # Remove the tree by replacing the tile with a walkable tile
                # We need to write to the current map's tile data
                # The simplest approach: set the tile in front to a path tile
                # This is a minimal intervention that the game handles correctly
                self.pyboy.memory[memory.ADDR_TILE_IN_FRONT] = 0x0C  # Path tile
                used.append("cut")

        # -----------------------------------------------------------------
        # Auto-Surf: Enable surfing when facing water
        # -----------------------------------------------------------------
        elif tile in memory.WATER_TILES:
            current_state = memory.read_byte(self.pyboy, memory.ADDR_WALK_BIKE_SURF)
            if current_state != 2 and memory.party_has_move(self.pyboy, memory.MOVE_SURF):
                # Set movement state to surfing
                self.pyboy.memory[memory.ADDR_WALK_BIKE_SURF] = 2
                used.append("surf")

        return used

    def get_taught_moves(self) -> list[str]:
        """
        Get list of all HM moves taught this episode.

        Returns:
            List of move names (e.g., ["cut", "surf"]).
        """
        return [HM_NAMES.get(m, f"move_{m}") for m in self.taught_moves]
