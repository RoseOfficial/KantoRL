"""
Reward functions for KantoRL.

This module provides Protocol-based reward functions that can be easily swapped,
understood, and extended. The reward system is a critical component of the
reinforcement learning pipeline, as it defines what behaviors the agent should
learn to maximize.

Architecture Role:
    The reward module sits between the environment (env.py) and the training
    loop (train.py). After each step, the environment captures a GameState
    snapshot and passes it to the configured RewardFunction to calculate
    the reward signal that guides learning.

    env.py (step) → GameState.from_pyboy() → RewardFunction.calculate() → reward

Design Philosophy:
    The module uses Python's Protocol pattern for structural subtyping, allowing
    any class that implements calculate(), reset(), and get_info() to be used
    as a reward function. This is more flexible than inheritance because:
    - No need to inherit from a base class
    - Duck typing allows easy testing with mock objects
    - New reward functions can be added without modifying existing code

Available Reward Functions:
    - DefaultReward: Multi-component reward matching PokemonRedExperiments V2
      (events, healing, badges, exploration, map discovery, stuck penalty)
    - BadgesOnlyReward: Minimal reward for testing (badges + step penalty)
    - ExplorationReward: Curiosity-driven reward (coordinates + maps only)

Reward Scaling:
    All rewards are scaled by a configurable reward_scale factor. This allows
    tuning the magnitude of rewards relative to:
    - The policy's learning rate
    - The discount factor (gamma)
    - The value function's scale

    Typical values: 0.1-1.0. Smaller values lead to more stable learning,
    larger values lead to faster but potentially unstable learning.

Usage:
    # Use the factory function to create rewards by name
    reward_fn = create_reward("default", reward_scale=0.5, explore_weight=0.1)

    # Or instantiate directly for more control
    reward_fn = DefaultReward(reward_scale=0.5, explore_weight=0.1)

    # In the environment step:
    state = GameState.from_pyboy(pyboy, step_count)
    reward = reward_fn.calculate(state, prev_state)
    info = reward_fn.get_info()  # For logging

Dependencies:
    - dataclasses: For clean state representation
    - typing: For Protocol and TYPE_CHECKING
    - numpy: For numerical operations (future use)
    - memory: For reading game state from PyBoy

References:
    - PokemonRedExperiments V2: https://github.com/PWhiddy/PokemonRedExperiments
    - Gymnasium reward design: https://gymnasium.farama.org/tutorials/reward_shaping/
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np  # noqa: F401 - Reserved for future numerical operations

if TYPE_CHECKING:
    # PyBoy is only needed for type hints, not at runtime
    # This avoids importing the heavy PyBoy module when not needed
    from pyboy import PyBoy

from kantorl import memory


# =============================================================================
# GAME STATE SNAPSHOT
# =============================================================================


@dataclass
class GameState:
    """
    Snapshot of relevant game state for reward calculation.

    This dataclass captures all the game information needed to calculate
    rewards at a given timestep. By creating immutable snapshots, we can
    easily compare current state to previous state for delta-based rewards.

    The snapshot is intentionally minimal - it only includes fields that
    are actually used by reward functions. Adding unused fields would waste
    memory when storing many states.

    Attributes:
        map_id: Current map ID (0-255). Maps are numbered areas in Pokemon Red
                like Pallet Town (0), Route 1 (12), etc. See global_map.py for
                the complete mapping.
        x: X coordinate within the current map (0-255). The coordinate system
           origin is the top-left corner of each map.
        y: Y coordinate within the current map (0-255). Y increases downward.
        badges: Number of gym badges collected (0-8). This is a key progress
                metric - the game is "won" after collecting all 8 badges and
                beating the Elite Four.
        event_count: Total number of event flags set in the game's memory.
                     Events track story progress, items collected, NPCs talked
                     to, etc. Higher count indicates more game progress.
        current_hp: Total current HP of all Pokemon in the party. Used for
                    healing rewards - we want to encourage the agent to heal.
        max_hp: Total maximum HP of all Pokemon in the party. Used to calculate
                HP ratio for normalized healing rewards.
        step_count: Number of environment steps taken this episode. Used for
                    debugging and time-based penalties if needed.

    Example:
        >>> state = GameState.from_pyboy(pyboy, step_count=1000)
        >>> print(f"At map {state.map_id} position ({state.x}, {state.y})")
        >>> print(f"HP: {state.current_hp}/{state.max_hp}")
        >>> print(f"Badges: {state.badges}, Events: {state.event_count}")

    Notes:
        - All numeric fields default to 0 for easy initialization
        - The from_pyboy classmethod is the preferred way to create states
        - States are immutable after creation (dataclass default)
    """

    # ===================
    # Position Information
    # ===================

    # Current map ID (see global_map.py for ID->name mapping)
    # Range: 0-255 (Pokemon Red has ~255 distinct map areas)
    map_id: int = 0

    # X coordinate within the map (column, increases rightward)
    # Range: 0-255 (most maps are much smaller than this)
    x: int = 0

    # Y coordinate within the map (row, increases downward)
    # Range: 0-255 (most maps are much smaller than this)
    y: int = 0

    # ===================
    # Progress Metrics
    # ===================

    # Number of gym badges collected
    # Range: 0-8 (Brock, Misty, Lt. Surge, Erika, Koga, Sabrina, Blaine, Giovanni)
    badges: int = 0

    # Count of event flags that have been triggered
    # Range: 0-2560 (total event flag bits in game memory)
    # Higher values indicate more story/game progress
    event_count: int = 0

    # ===================
    # Party Health
    # ===================

    # Total current HP across all party Pokemon
    # Range: 0-~3000+ (depends on party size and levels)
    current_hp: int = 0

    # Total maximum HP across all party Pokemon
    # Range: 0-~3000+ (used to calculate HP ratio)
    max_hp: int = 0

    # ===================
    # Episode Tracking
    # ===================

    # Number of environment steps taken this episode
    # Used for debugging and potential time-based rewards/penalties
    step_count: int = 0

    @classmethod
    def from_pyboy(cls, pyboy: "PyBoy", step_count: int = 0) -> "GameState":
        """
        Create a state snapshot from a PyBoy emulator instance.

        This is the primary way to create GameState objects. It reads all
        relevant information from the emulator's memory in a single pass,
        creating an immutable snapshot that can be used for reward calculation.

        Args:
            pyboy: The PyBoy emulator instance to read state from.
                   Must have a Pokemon Red ROM loaded and running.
            step_count: Current step count for this episode. Defaults to 0.
                       Pass the environment's internal step counter here.

        Returns:
            A new GameState instance populated with current game values.

        Example:
            >>> state = GameState.from_pyboy(env.pyboy, env.step_count)
            >>> reward = reward_fn.calculate(state, prev_state)

        Notes:
            - Memory reads are relatively fast (~microseconds per read)
            - Position is read as a tuple (map_id, x, y) from memory.get_position
            - HP is read as a tuple (current, max) from memory.get_total_party_hp
            - Badges and events are read individually
        """
        # Read position as a tuple: (map_id, x, y)
        # This is a common operation so it's bundled in memory.py
        map_id, x, y = memory.get_position(pyboy)

        # Read party HP totals: (current_hp, max_hp)
        # Used for healing rewards
        current_hp, max_hp = memory.get_total_party_hp(pyboy)

        return cls(
            map_id=map_id,
            x=x,
            y=y,
            badges=memory.get_badges(pyboy),  # Count of badges (0-8)
            event_count=memory.count_event_flags(pyboy),  # Total events triggered
            current_hp=current_hp,
            max_hp=max_hp,
            step_count=step_count,
        )


# =============================================================================
# REWARD FUNCTION PROTOCOL
# =============================================================================


class RewardFunction(Protocol):
    """
    Protocol defining the interface for reward functions.

    This Protocol uses Python's structural subtyping to define what methods
    a reward function must implement. Any class that has these methods with
    compatible signatures can be used as a RewardFunction, without needing
    to inherit from a base class.

    Protocol Benefits:
        - Decoupling: Reward functions don't depend on a base class
        - Testability: Easy to create mock reward functions for testing
        - Flexibility: External code can provide custom rewards
        - Documentation: The Protocol serves as an interface contract

    Required Methods:
        calculate(): Compute the reward for a state transition
        reset(): Clear internal state for a new episode
        get_info(): Return debugging/logging information

    Example Implementation:
        >>> @dataclass
        ... class MyReward:
        ...     total: float = 0.0
        ...
        ...     def calculate(self, state: GameState, prev_state: GameState | None) -> float:
        ...         reward = 1.0 if state.badges > (prev_state.badges if prev_state else 0) else 0.0
        ...         self.total += reward
        ...         return reward
        ...
        ...     def reset(self) -> None:
        ...         self.total = 0.0
        ...
        ...     def get_info(self) -> dict:
        ...         return {"total_reward": self.total}

    Notes:
        - Protocols are a Python 3.8+ feature (typing.Protocol)
        - Type checkers use structural subtyping to verify compatibility
        - At runtime, no inheritance check is performed (duck typing)
    """

    def calculate(self, state: GameState, prev_state: GameState | None) -> float:
        """
        Calculate the reward for transitioning to the current state.

        This is the core method of any reward function. It examines the
        current game state and optionally the previous state to compute
        a scalar reward signal that guides the agent's learning.

        Args:
            state: The current game state after the action was taken.
            prev_state: The previous game state before the action, or None
                       if this is the first step of an episode.

        Returns:
            A float reward value. Positive values encourage the behavior
            that led to this state, negative values discourage it.
            The magnitude indicates the strength of the signal.

        Notes:
            - Called once per environment step
            - Should be deterministic given the same inputs
            - May update internal state (e.g., visited coordinates)
            - prev_state is None on the first step after reset()
        """
        ...

    def reset(self) -> None:
        """
        Reset internal state for a new episode.

        Called by the environment when reset() is invoked. This should
        clear any episode-specific tracking (visited coordinates, max
        values, etc.) while preserving configuration (reward_scale, etc.).

        Notes:
            - Called before the first step of each episode
            - Must be idempotent (safe to call multiple times)
            - Should not return anything
        """
        ...

    def get_info(self) -> dict:
        """
        Get reward breakdown and statistics for logging.

        Returns a dictionary of information useful for debugging and
        monitoring training progress. This is included in the info dict
        returned by env.step().

        Returns:
            A dictionary containing:
            - "reward_breakdown": Dict of individual reward components
            - Any additional statistics (unique coords, max badges, etc.)

        Notes:
            - Called after calculate() in each step
            - Should not modify internal state
            - Values should be JSON-serializable for TensorBoard logging
        """
        ...


# =============================================================================
# DEFAULT REWARD (MATCHING V2)
# =============================================================================


@dataclass
class DefaultReward:
    """
    Default multi-component reward function matching PokemonRedExperiments V2.

    This reward function combines multiple signals to encourage diverse behaviors:
    - Event progress: Rewards triggering story events and collecting items
    - Healing: Rewards restoring Pokemon HP (encourages using Pokemon Centers)
    - Badges: Large rewards for defeating gym leaders
    - Exploration: Small rewards for visiting new map coordinates
    - Map discovery: Medium rewards for entering new map areas
    - Stuck penalty: Small negative reward when making no progress

    The multi-component design helps balance exploration vs exploitation and
    prevents the agent from getting stuck optimizing a single metric.

    Attributes:
        reward_scale: Global multiplier for all rewards. Typical range: 0.1-1.0.
                      Lower values lead to more stable but slower learning.
        explore_weight: Additional multiplier for exploration rewards.
                       Higher values encourage more map coverage.

        visited_coords: Set of (map_id, x, y) tuples visited this episode.
                       Used to calculate exploration rewards.
        visited_maps: Set of map_ids visited this episode.
                     Used to calculate map discovery rewards.
        max_event_count: Highest event count seen this episode.
                        Used for delta-based event rewards.
        max_badges: Highest badge count seen this episode.
                   Used for delta-based badge rewards.
        total_healing: Cumulative healing done this episode (for logging).
        steps_without_progress: Counter for stuck detection.

        last_rewards: Dictionary of most recent reward components (for logging).

    Reward Magnitudes (before scaling):
        - Event: 8.0 per new event
        - Healing: 10.0 × HP_ratio_increase
        - Badge: 10.0 per new badge
        - Exploration: 0.01 per new coordinate (0.1 × explore_weight × 0.1)
        - Map discovery: 2.0 per new map
        - Stuck penalty: -0.05 after 500 steps without progress

    Example:
        >>> reward_fn = DefaultReward(reward_scale=0.5, explore_weight=0.1)
        >>> reward = reward_fn.calculate(state, prev_state)
        >>> print(reward_fn.get_info()["reward_breakdown"])
        {'event': 0.0, 'heal': 0.0, 'badge': 0.0, 'explore': 0.005, ...}

    Notes:
        - Rewards are cumulative within an episode (via tracking sets)
        - The stuck penalty only triggers after 500+ steps without any reward
        - Badge and event rewards are based on delta, not absolute values
    """

    # ===================
    # Configuration
    # ===================

    # Global reward multiplier - scales all reward components
    # Typical range: 0.1-1.0. Lower = more stable, higher = faster learning
    reward_scale: float = 0.5

    # Additional weight for exploration rewards
    # Higher values encourage more thorough map coverage
    explore_weight: float = 0.1

    # ===================
    # Episode Tracking
    # ===================

    # Set of (map_id, x, y) coordinates visited this episode
    # Used for exploration rewards - only new coordinates give reward
    visited_coords: set = field(default_factory=set)

    # Set of map_id values visited this episode
    # Used for map discovery rewards - entering new areas
    visited_maps: set = field(default_factory=set)

    # Highest event count seen this episode
    # Events only reward when count increases (delta-based)
    max_event_count: int = 0

    # Highest badge count seen this episode
    # Badges only reward when count increases (delta-based)
    max_badges: int = 0

    # Cumulative HP restoration this episode (for logging)
    # Tracks total healing to monitor Pokemon Center usage
    total_healing: float = 0.0

    # Counter for steps without any positive reward
    # Triggers stuck penalty after threshold exceeded
    steps_without_progress: int = 0

    # ===================
    # Logging
    # ===================

    # Dictionary of most recent reward components
    # Used for detailed logging and debugging
    last_rewards: dict = field(default_factory=dict)

    def calculate(self, state: GameState, prev_state: GameState | None) -> float:
        """
        Calculate the multi-component reward for the current state.

        Evaluates six reward components:
        1. Event progress - new story/item events triggered
        2. Healing - HP restoration (Pokemon Center usage)
        3. Badge collection - gym victories
        4. Exploration - visiting new coordinates
        5. Map discovery - entering new map areas
        6. Stuck penalty - negative reward for lack of progress

        Args:
            state: Current game state after the action.
            prev_state: Previous game state, or None if first step.

        Returns:
            Total reward as sum of all components (scaled by reward_scale).

        Notes:
            - Updates internal tracking state (visited sets, max values)
            - Stores component breakdown in self.last_rewards for logging
            - Stuck penalty only activates after 500+ steps without progress
        """
        # Initialize reward components dictionary
        # Each component starts at 0 and is set if its condition is met
        rewards = {
            "event": 0.0,       # Story/item event progress
            "heal": 0.0,        # HP restoration
            "badge": 0.0,       # Gym badge collection
            "explore": 0.0,     # New coordinate visited
            "map_progress": 0.0,  # New map area entered
            "stuck": 0.0,       # Penalty for no progress
        }

        # ---------------------------------------------------------------------
        # Event Progress Reward
        # ---------------------------------------------------------------------
        # Reward for triggering new events (story progress, items, etc.)
        # Events are tracked by the game's event flag system
        if state.event_count > self.max_event_count:
            # Calculate number of new events triggered
            new_events = state.event_count - self.max_event_count
            # Reward: 8.0 per event × scale (significant progress signal)
            rewards["event"] = self.reward_scale * new_events * 8.0
            # Update max to only reward new events
            self.max_event_count = state.event_count

        # ---------------------------------------------------------------------
        # Healing Reward
        # ---------------------------------------------------------------------
        # Reward for restoring Pokemon HP (encourages Pokemon Center usage)
        # Only applies if we have a previous state to compare against
        if prev_state is not None and state.max_hp > 0:
            # Calculate HP as a ratio (0.0 to 1.0) for normalized comparison
            hp_ratio_now = state.current_hp / state.max_hp
            hp_ratio_prev = (
                prev_state.current_hp / prev_state.max_hp if prev_state.max_hp > 0 else 0
            )
            # Only reward HP increases (healing), not decreases (damage)
            if hp_ratio_now > hp_ratio_prev:
                healing = hp_ratio_now - hp_ratio_prev
                # Reward: 10.0 × healing_ratio × scale
                rewards["heal"] = self.reward_scale * healing * 10.0
                # Track total healing for logging
                self.total_healing += healing

        # ---------------------------------------------------------------------
        # Badge Collection Reward
        # ---------------------------------------------------------------------
        # Big reward for defeating gym leaders and collecting badges
        # This is the primary progress metric in Pokemon Red
        if state.badges > self.max_badges:
            # Calculate number of new badges (usually 1)
            new_badges = state.badges - self.max_badges
            # Reward: 10.0 per badge × scale (major milestone)
            rewards["badge"] = self.reward_scale * new_badges * 10.0
            # Update max to only reward new badges
            self.max_badges = state.badges

        # ---------------------------------------------------------------------
        # Exploration Reward
        # ---------------------------------------------------------------------
        # Small reward for visiting new coordinates
        # Encourages thorough exploration of each map
        coord = (state.map_id, state.x, state.y)
        if coord not in self.visited_coords:
            self.visited_coords.add(coord)
            # Reward: 0.01 × scale (small but cumulative)
            # explore_weight allows tuning exploration vs exploitation
            rewards["explore"] = self.reward_scale * self.explore_weight * 0.1

        # ---------------------------------------------------------------------
        # Map Discovery Reward
        # ---------------------------------------------------------------------
        # Medium reward for entering a new map area
        # Encourages moving between different game locations
        if state.map_id not in self.visited_maps:
            self.visited_maps.add(state.map_id)
            # Reward: 2.0 × scale (discovering new areas is valuable)
            rewards["map_progress"] = self.reward_scale * 2.0

        # ---------------------------------------------------------------------
        # Stuck Penalty
        # ---------------------------------------------------------------------
        # Small negative reward if making no progress for extended period
        # Prevents the agent from getting stuck in unproductive loops
        has_progress = any(r > 0 for r in rewards.values())
        if has_progress:
            # Reset counter when any progress is made
            self.steps_without_progress = 0
        else:
            # Increment counter for steps without reward
            self.steps_without_progress += 1
            # Apply penalty after threshold (500 steps = ~8 seconds of gameplay)
            if self.steps_without_progress > 500:
                # Penalty: -0.05 × scale (gentle nudge to try something different)
                rewards["stuck"] = self.reward_scale * -0.05

        # Store breakdown for logging via get_info()
        self.last_rewards = rewards

        # Return sum of all reward components
        return sum(rewards.values())

    def reset(self) -> None:
        """
        Reset internal state for a new episode.

        Clears all episode-specific tracking while preserving configuration.
        Called by the environment when env.reset() is invoked.

        Resets:
            - visited_coords: Empty set (all coordinates are "new" again)
            - visited_maps: Empty set (all maps are "new" again)
            - max_event_count: 0 (reset delta tracking)
            - max_badges: 0 (reset delta tracking)
            - total_healing: 0.0 (reset logging counter)
            - steps_without_progress: 0 (reset stuck detection)
            - last_rewards: Empty dict (clear logging)

        Notes:
            - Configuration (reward_scale, explore_weight) is preserved
            - Safe to call multiple times (idempotent)
        """
        self.visited_coords.clear()
        self.visited_maps.clear()
        self.max_event_count = 0
        self.max_badges = 0
        self.total_healing = 0.0
        self.steps_without_progress = 0
        self.last_rewards = {}

    def get_info(self) -> dict:
        """
        Get reward breakdown and episode statistics for logging.

        Returns detailed information about the reward calculation and
        episode progress. This is included in the info dict from env.step()
        and can be logged to TensorBoard for analysis.

        Returns:
            Dictionary containing:
            - reward_breakdown: Dict of individual reward components
            - unique_coords: Number of unique coordinates visited
            - unique_maps: Number of unique maps visited
            - max_badges: Highest badge count this episode
            - max_events: Highest event count this episode
            - total_healing: Cumulative HP restoration this episode

        Example:
            >>> info = reward_fn.get_info()
            >>> print(f"Visited {info['unique_coords']} tiles in {info['unique_maps']} maps")
        """
        return {
            "reward_breakdown": self.last_rewards.copy(),
            "unique_coords": len(self.visited_coords),
            "unique_maps": len(self.visited_maps),
            "max_badges": self.max_badges,
            "max_events": self.max_event_count,
            "total_healing": self.total_healing,
        }


# =============================================================================
# SIMPLE REWARD VARIANTS
# =============================================================================


@dataclass
class BadgesOnlyReward:
    """
    Minimal reward function: badges only with step penalty.

    This simple reward is useful for:
    - Testing the training pipeline without complex rewards
    - Debugging environment issues
    - Establishing baseline performance
    - Fast iteration during development

    The step penalty (-0.001 per step) encourages efficient behavior and
    prevents the agent from idling. Without it, the agent might learn to
    do nothing while waiting for random badge opportunities.

    Attributes:
        reward_scale: Multiplier for badge rewards. Default 1.0.
        max_badges: Highest badge count seen this episode.
        last_rewards: Most recent reward breakdown (for logging).

    Reward Structure:
        - Badge: +1.0 × reward_scale per new badge
        - Step: -0.001 per step (encourages efficiency)

    Example:
        >>> reward_fn = BadgesOnlyReward(reward_scale=1.0)
        >>> # Getting a badge gives +1.0, each step costs -0.001
        >>> # Net positive after ~1000 steps if a badge is collected

    Notes:
        - Much simpler than DefaultReward (easier to debug)
        - No exploration incentive (agent may get stuck)
        - Good for testing but not recommended for real training
    """

    # Multiplier for badge rewards
    reward_scale: float = 1.0

    # Highest badge count seen (for delta rewards)
    max_badges: int = 0

    # Most recent reward breakdown (for logging)
    last_rewards: dict = field(default_factory=dict)

    def calculate(self, state: GameState, prev_state: GameState | None) -> float:
        """
        Calculate badge-only reward with step penalty.

        Simple reward: +reward_scale per new badge, -0.001 per step.

        Args:
            state: Current game state.
            prev_state: Previous state (unused in this simple reward).

        Returns:
            Reward value (positive for badges, negative step penalty).
        """
        # Initialize with constant step penalty
        # This encourages the agent to achieve goals quickly
        rewards = {"badge": 0.0, "step": -0.001}

        # Check for new badges (delta-based reward)
        if state.badges > self.max_badges:
            new_badges = state.badges - self.max_badges
            rewards["badge"] = self.reward_scale * new_badges
            self.max_badges = state.badges

        self.last_rewards = rewards
        return sum(rewards.values())

    def reset(self) -> None:
        """Reset for new episode."""
        self.max_badges = 0
        self.last_rewards = {}

    def get_info(self) -> dict:
        """Get reward breakdown for logging."""
        return {"reward_breakdown": self.last_rewards.copy(), "max_badges": self.max_badges}


@dataclass
class ExplorationReward:
    """
    Exploration-focused reward for curiosity-driven learning.

    This reward function emphasizes map discovery and coordinate exploration
    over events and badges. It's useful for:
    - Training agents that thoroughly explore the game world
    - Research on intrinsic motivation and curiosity
    - Creating agents that find hidden items/areas
    - Pre-training before switching to goal-directed rewards

    Unlike DefaultReward, this function ignores story progress (events, badges)
    and only rewards visiting new locations.

    Attributes:
        reward_scale: Multiplier for all rewards. Default 0.5.
        visited_coords: Set of (map_id, x, y) tuples visited.
        visited_maps: Set of map_ids visited.
        last_rewards: Most recent reward breakdown (for logging).

    Reward Structure:
        - Coordinate: +0.05 per new (map_id, x, y) coordinate
        - Map: +0.5 per new map area entered

    Example:
        >>> reward_fn = ExplorationReward(reward_scale=0.5)
        >>> # Visiting 100 new tiles gives +5.0 total
        >>> # Entering 5 new maps gives +2.5 total

    Notes:
        - No event/badge rewards (pure exploration)
        - Good for building map coverage before goal-directed training
        - May explore thoroughly but never beat the game
    """

    # Multiplier for exploration rewards
    reward_scale: float = 0.5

    # Set of (map_id, x, y) coordinates visited this episode
    visited_coords: set = field(default_factory=set)

    # Set of map_ids visited this episode
    visited_maps: set = field(default_factory=set)

    # Most recent reward breakdown (for logging)
    last_rewards: dict = field(default_factory=dict)

    def calculate(self, state: GameState, prev_state: GameState | None) -> float:
        """
        Calculate exploration-focused reward.

        Rewards visiting new coordinates and discovering new map areas.
        No rewards for events, badges, or other game progress.

        Args:
            state: Current game state.
            prev_state: Previous state (unused in this reward).

        Returns:
            Exploration reward (coordinate + map discovery).
        """
        rewards = {"explore": 0.0, "map": 0.0}

        # Reward for visiting new coordinates
        coord = (state.map_id, state.x, state.y)
        if coord not in self.visited_coords:
            self.visited_coords.add(coord)
            # +0.1 × reward_scale per new coordinate
            rewards["explore"] = self.reward_scale * 0.1

        # Reward for entering new map areas
        if state.map_id not in self.visited_maps:
            self.visited_maps.add(state.map_id)
            # +1.0 × reward_scale per new map
            rewards["map"] = self.reward_scale * 1.0

        self.last_rewards = rewards
        return sum(rewards.values())

    def reset(self) -> None:
        """Reset for new episode."""
        self.visited_coords.clear()
        self.visited_maps.clear()
        self.last_rewards = {}

    def get_info(self) -> dict:
        """Get exploration statistics for logging."""
        return {
            "reward_breakdown": self.last_rewards.copy(),
            "unique_coords": len(self.visited_coords),
            "unique_maps": len(self.visited_maps),
        }


# =============================================================================
# REWARD FACTORY
# =============================================================================

# Registry of available reward functions
# Maps string names to their implementing classes
REWARD_FUNCTIONS = {
    "default": DefaultReward,      # Multi-component V2 reward
    "badges_only": BadgesOnlyReward,  # Minimal testing reward
    "exploration": ExplorationReward,  # Curiosity-driven reward
}


def create_reward(
    name: str = "default",
    reward_scale: float = 0.5,
    explore_weight: float = 0.1,
) -> RewardFunction:
    """
    Create a reward function by name using the factory pattern.

    This is the recommended way to instantiate reward functions, as it
    provides a clean interface for configuration files and command-line
    arguments.

    Args:
        name: Name of the reward function to create. Must be one of:
              - "default": Multi-component V2 reward (recommended)
              - "badges_only": Minimal testing reward
              - "exploration": Curiosity-driven reward
        reward_scale: Global multiplier for all rewards. Typical range 0.1-1.0.
                     Lower values lead to more stable learning.
        explore_weight: Weight for exploration bonus in DefaultReward.
                       Only used when name="default". Higher values
                       encourage more thorough map coverage.

    Returns:
        A reward function instance implementing the RewardFunction protocol.

    Raises:
        ValueError: If the reward name is not recognized.

    Example:
        >>> # Create default reward for normal training
        >>> reward = create_reward("default", reward_scale=0.5)

        >>> # Create exploration reward for curiosity-driven learning
        >>> reward = create_reward("exploration", reward_scale=1.0)

        >>> # Create minimal reward for testing
        >>> reward = create_reward("badges_only")

    Notes:
        - explore_weight is only used by DefaultReward
        - All reward functions receive reward_scale
        - New reward types can be added to REWARD_FUNCTIONS dict
    """
    # Validate reward name
    if name not in REWARD_FUNCTIONS:
        available = list(REWARD_FUNCTIONS.keys())
        raise ValueError(f"Unknown reward: {name}. Available: {available}")

    # Get the reward class
    cls = REWARD_FUNCTIONS[name]

    # Create instance with appropriate arguments
    # Only DefaultReward uses explore_weight
    if name == "default":
        return cls(reward_scale=reward_scale, explore_weight=explore_weight)
    return cls(reward_scale=reward_scale)
