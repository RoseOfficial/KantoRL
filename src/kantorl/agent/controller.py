"""
Scripted button sequence controller for KantoRL agent.

This module translates high-level logical actions (like "use move 2" or
"run from battle") into sequences of GameBoy button presses. The GameBoy
has no mouse or touch input — everything is done through directional pad
and A/B/Start/Select buttons, so navigating menus requires precise
multi-button sequences.

Architecture Role:
    The controller sits between the battler/manager decision modules and
    the actual environment step. When a module decides "use super-effective
    move in slot 3", the controller queues the button presses needed to
    navigate the battle menu and select that move.

    Battler → "FIGHT_3" → Controller → [A, DOWN, DOWN, A] → env.step()

Design Decisions:
    - Queue-based: Sequences are queued and executed one button per step.
      This is necessary because each env.step() only accepts one action.
    - No memory reading: The controller is stateless regarding game state.
      It trusts that the caller (AgentWrapper) only triggers sequences
      at appropriate times (e.g., battle menu is open).
    - Fail-safe: If a sequence doesn't work (menu state mismatch), the
      worst case is a few wasted button presses — the agent recovers.

Action Index Reference (matching env.py):
    0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START

Dependencies:
    None (pure logic module, no external imports)
"""

from __future__ import annotations

from enum import IntEnum

# =============================================================================
# ACTION CONSTANTS
# =============================================================================
# These match the action indices defined in env.py PRESS_ACTIONS.
# Duplicated here to avoid circular imports — controller is a leaf module.

class Action(IntEnum):
    """GameBoy action indices matching env.py action space."""

    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    A = 5
    B = 6
    START = 7


# =============================================================================
# BATTLE ACTION ENUM
# =============================================================================


class BattleAction(IntEnum):
    """
    Logical battle actions that the battler module can request.

    These are translated into button sequences by the MenuController.
    FIGHT_1..FIGHT_4 correspond to the four move slots.
    SWITCH_1..SWITCH_6 correspond to party positions.
    """

    FIGHT_1 = 0
    FIGHT_2 = 1
    FIGHT_3 = 2
    FIGHT_4 = 3
    SWITCH_1 = 4
    SWITCH_2 = 5
    SWITCH_3 = 6
    SWITCH_4 = 7
    SWITCH_5 = 8
    SWITCH_6 = 9
    RUN = 10


# =============================================================================
# BUTTON SEQUENCE
# =============================================================================


class ButtonSequence:
    """
    A queue of button presses to execute one per env step.

    Tracks a list of action indices with a cursor. Each call to
    next_action() returns the next button and advances the cursor.

    Attributes:
        actions: List of action indices to execute in order.
        cursor: Current position in the sequence.
    """

    __slots__ = ("actions", "cursor")

    def __init__(self, actions: list[int]) -> None:
        """
        Create a button sequence.

        Args:
            actions: List of action indices (0-7) to execute in order.
        """
        self.actions = actions
        self.cursor = 0

    def next_action(self) -> int:
        """
        Get the next action in the sequence and advance cursor.

        Returns:
            Action index (0-7). Returns NOOP if sequence is complete.
        """
        if self.cursor < len(self.actions):
            action = self.actions[self.cursor]
            self.cursor += 1
            return action
        return Action.NOOP

    def is_complete(self) -> bool:
        """Check if all actions in the sequence have been executed."""
        return self.cursor >= len(self.actions)


# =============================================================================
# MENU CONTROLLER
# =============================================================================


class MenuController:
    """
    Translates logical actions into GameBoy button sequences.

    Maintains a queue of ButtonSequence objects. When the AgentWrapper
    asks for an action, the controller returns the next button from
    the current sequence. When a sequence completes, it's discarded
    and the next one starts.

    The controller is pickle-safe (no file handles, threads, or lambdas)
    for SubprocVecEnv compatibility.

    Example:
        >>> ctrl = MenuController()
        >>> ctrl.start_battle_move(2)  # Select move in slot 3
        >>> while ctrl.has_pending_sequence():
        ...     action = ctrl.get_next_action()
        ...     obs, reward, done, trunc, info = env.step(action)
    """

    __slots__ = ("_sequence",)

    def __init__(self) -> None:
        """Initialize controller with no pending sequences."""
        self._sequence: ButtonSequence | None = None

    def reset(self) -> None:
        """Clear any pending button sequences. Called on episode reset."""
        self._sequence = None

    def has_pending_sequence(self) -> bool:
        """Check if there are button presses waiting to be executed."""
        return self._sequence is not None and not self._sequence.is_complete()

    def get_next_action(self) -> int:
        """
        Get the next button press from the current sequence.

        Returns:
            Action index (0-7). Returns NOOP if no sequence is pending.
        """
        if self._sequence is None or self._sequence.is_complete():
            self._sequence = None
            return Action.NOOP
        action = self._sequence.next_action()
        # Auto-clear completed sequences
        if self._sequence.is_complete():
            self._sequence = None
        return action

    # =========================================================================
    # BATTLE MENU SEQUENCES
    # =========================================================================

    def start_battle_move(self, slot: int) -> None:
        """
        Queue button presses to select a battle move by slot index.

        Battle menu layout:
            FIGHT  BAG
            PKMN   RUN

        FIGHT submenu:
            Move 1  Move 2
            Move 3  Move 4

        Args:
            slot: Move slot index (0-3).

        Sequence:
            1. A to enter FIGHT menu
            2. Navigate to the correct slot (DOWN/RIGHT)
            3. A to confirm move selection
        """
        # Start with A to enter FIGHT (cursor defaults to FIGHT)
        actions: list[int] = [Action.A]

        # Navigate to the correct move slot
        # Move layout is a 2x2 grid:
        #   Slot 0 (top-left)  |  Slot 1 (top-right)
        #   Slot 2 (bottom-left) | Slot 3 (bottom-right)
        if slot == 1:
            actions.append(Action.RIGHT)
        elif slot == 2:
            actions.append(Action.DOWN)
        elif slot == 3:
            actions.append(Action.DOWN)
            actions.append(Action.RIGHT)
        # slot 0 needs no navigation (cursor starts there)

        # Confirm selection
        actions.append(Action.A)

        self._sequence = ButtonSequence(actions)

    def start_battle_run(self) -> None:
        """
        Queue button presses to attempt running from battle.

        Battle menu layout:
            FIGHT  BAG
            PKMN   RUN

        Sequence: DOWN to go to bottom row, RIGHT to RUN, A to confirm.
        """
        self._sequence = ButtonSequence([
            Action.DOWN,   # Move to bottom row
            Action.RIGHT,  # Move to RUN
            Action.A,      # Confirm RUN
        ])

    def start_battle_switch(self, party_slot: int) -> None:
        """
        Queue button presses to switch to a different party Pokemon.

        Battle menu layout:
            FIGHT  BAG
            PKMN   RUN

        Args:
            party_slot: Party position to switch to (0-5).

        Sequence:
            1. DOWN to PKMN row
            2. A to enter party menu
            3. DOWN × party_slot to navigate to Pokemon
            4. A to select
            5. A to confirm SWITCH
        """
        actions: list[int] = [
            Action.DOWN,  # Move to PKMN row
            Action.A,     # Enter party menu
        ]

        # Navigate down to the target party slot
        for _ in range(party_slot):
            actions.append(Action.DOWN)

        # Select Pokemon and confirm switch
        actions.append(Action.A)  # Select the Pokemon
        actions.append(Action.A)  # Confirm SWITCH option

        self._sequence = ButtonSequence(actions)

    def start_dialogue_advance(self, presses: int = 3) -> None:
        """
        Queue A button presses to advance through dialogue text.

        NPC dialogue, battle text, and other messages require pressing
        A (or B) to advance. We press A multiple times to get through
        multi-page text boxes.

        Args:
            presses: Number of A presses to queue. Default 3 handles
                    most single-interaction dialogues.
        """
        self._sequence = ButtonSequence([Action.A] * presses)

    def start_from_battle_action(self, battle_action: BattleAction) -> None:
        """
        Queue the appropriate button sequence for a BattleAction.

        This is the main entry point used by the AgentWrapper to convert
        the battler's decision into button presses.

        Args:
            battle_action: Logical battle action from the battler module.
        """
        if battle_action == BattleAction.RUN:
            self.start_battle_run()
        elif battle_action.value <= BattleAction.FIGHT_4.value:
            # FIGHT_1 through FIGHT_4
            self.start_battle_move(battle_action.value)
        else:
            # SWITCH_1 through SWITCH_6
            party_slot = battle_action.value - BattleAction.SWITCH_1.value
            self.start_battle_switch(party_slot)
