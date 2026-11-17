"""
KantoRL main Gymnasium environment.

This module provides the core reinforcement learning environment for Pokemon Red.
It wraps the PyBoy GameBoy emulator to create a Gymnasium-compatible environment
that can be used with stable-baselines3 and other RL libraries.

Architecture Role:
    KantoRedEnv is the central component of KantoRL. It:
    - Manages the PyBoy emulator instance
    - Translates agent actions to button presses
    - Extracts observations from game memory and screen
    - Calculates rewards using pluggable reward functions
    - Tracks exploration progress

Key Design Decisions:
    - Lazy emulator initialization: PyBoy is only created when first accessed
    - Dict observation space: Combines screen, game state, and history
    - Frame stacking: Multiple frames for temporal information
    - Fourier-encoded levels: Smooth representation for value function learning
    - Exploration bitmap: 48x48 downscaled global map of visited locations

Observation Space Components:
    Selected by ``KantoConfig.obs_components``. The default reproduces the
    historical observation exactly; omit entries to run a leaner tier.
    - screens: (3, 72, 80) stacked grayscale frames for visual input
    - health: (1,) party HP fraction [0, 1] for survival awareness
    - level: (8,) Fourier-encoded party levels for progress tracking
    - badges: (8,) binary gym badge flags for milestone tracking
    - events: (2560,) binary event flags for fine-grained progress
    - map: (48, 48) exploration bitmap for spatial memory
    - recent_actions: (3,) last actions for action history

Action Space:
    Discrete(8) with actions:
    0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START

Usage:
    >>> from kantorl.env import KantoRedEnv
    >>> env = KantoRedEnv(rom_path="pokemon_red.gb")
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(1)  # Press UP
    >>> env.close()

Dependencies:
    - gymnasium: For the Gym environment interface
    - pyboy: GameBoy emulator for running Pokemon Red
    - numpy: For array operations
    - cv2 (opencv): For screen grayscale conversion and resizing
    - kantorl.memory: For reading game state from memory
    - kantorl.rewards: For reward calculation
    - kantorl.global_map: For coordinate conversion
"""

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from kantorl import memory
from kantorl.config import KantoConfig
from kantorl.global_map import GLOBAL_MAP_HEIGHT, GLOBAL_MAP_WIDTH, local_to_global
from kantorl.obs import (
    ACTION_HISTORY_LEN,
    EVENT_OBS_DIM,
    MAP_OBS_SHAPE,
    ObsContext,
    build_producers,
)
from kantorl.rewards import GameState, RewardFunction, create_reward

# =============================================================================
# NAMED EVENT MILESTONE MAP
# =============================================================================
# Maps specific event flag bit indices to human-readable milestone names.
# Loaded lazily from data/events.json (80 curated milestones).
# Used to decode opaque event counts into meaningful progress indicators.

_MILESTONE_MAP: dict[int, str] | None = None


def _load_milestones() -> dict[int, str]:
    """
    Load named event milestones from the JSON data file.

    Implements lazy loading with module-level caching — the file is read
    once per process (important since SubprocVecEnv forks per env).

    Returns:
        Dictionary mapping event flag bit indices to milestone names.
        Returns empty dict if the data file is not found.
    """
    global _MILESTONE_MAP
    if _MILESTONE_MAP is None:
        data_path = Path(__file__).parent / "data" / "events.json"
        if data_path.exists():
            with open(data_path) as f:
                raw = json.load(f)
            # JSON keys are strings — convert to int for direct array indexing
            _MILESTONE_MAP = {int(k): v for k, v in raw.items()}
        else:
            _MILESTONE_MAP = {}
    return _MILESTONE_MAP


# =============================================================================
# ACTION MAPPING CONSTANTS
# =============================================================================
# Maps discrete action indices to PyBoy WindowEvent button presses.
# PyBoy uses WindowEvent for reliable input handling across platforms.

# Button press events for each action
# Index 0 is NOOP (no button pressed), indices 1-7 are actual buttons
# Using WindowEvent ensures proper timing and input registration
PRESS_ACTIONS = [
    None,  # Action 0: NOOP - do nothing
    WindowEvent.PRESS_ARROW_UP,  # Action 1: Move up
    WindowEvent.PRESS_ARROW_DOWN,  # Action 2: Move down
    WindowEvent.PRESS_ARROW_LEFT,  # Action 3: Move left
    WindowEvent.PRESS_ARROW_RIGHT,  # Action 4: Move right
    WindowEvent.PRESS_BUTTON_A,  # Action 5: A button (confirm/interact)
    WindowEvent.PRESS_BUTTON_B,  # Action 6: B button (cancel/run)
    WindowEvent.PRESS_BUTTON_START,  # Action 7: Start (menu)
]

# Button release events corresponding to each press
# Must release buttons after pressing to avoid stuck inputs
RELEASE_ACTIONS = [
    None,  # Action 0: NOOP - nothing to release
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

# Human-readable action names for debugging and visualization
ACTION_NAMES = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]


# =============================================================================
# MAIN ENVIRONMENT CLASS
# =============================================================================


class KantoRedEnv(gym.Env):
    """
    Pokemon Red environment for reinforcement learning.

    This environment wraps the PyBoy GameBoy emulator to provide a standard
    Gymnasium interface for training RL agents to play Pokemon Red.

    The observation space is a dictionary containing multiple components:
    - screens: Stacked grayscale frames for visual input
    - health: Party HP fraction for survival awareness
    - level: Fourier-encoded party levels for smooth learning
    - badges: Gym badges obtained as progress milestones
    - events: Game event flags for fine-grained progress
    - map: Exploration bitmap showing visited areas
    - recent_actions: History of recent actions

    The action space is Discrete(8) corresponding to GameBoy buttons:
    0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START

    Attributes:
        config (KantoConfig): Configuration parameters.
        reward_fn (RewardFunction): Reward calculation function.
        render_mode (str | None): Rendering mode ('human', 'rgb_array', None).
        step_count (int): Steps taken in current episode.
        prev_state (GameState | None): Previous game state for reward calculation.
        recent_actions (np.ndarray): Last 3 actions taken.
        explore_map (np.ndarray): 48x48 bitmap of visited global coordinates.
        frame_stack (np.ndarray): Stacked grayscale frames.

    Example:
        >>> env = KantoRedEnv(rom_path="pokemon_red.gb")
        >>> obs, info = env.reset()
        >>> for _ in range(1000):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     if terminated or truncated:
        ...         obs, info = env.reset()
        >>> env.close()

    Notes:
        - The emulator is lazily initialized on first access
        - Reset loads from a save state for consistent starting point
        - Frame skipping (action_freq) controls game speed vs control granularity
    """

    # Gymnasium metadata for rendering
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: KantoConfig | None = None,
        rom_path: str | None = None,
        reward_fn: RewardFunction | str = "default",
        render_mode: str | None = None,
        initial_step_offset: int = 0,
    ):
        """
        Initialize the Pokemon Red environment.

        Args:
            config: Configuration object with hyperparameters. If None, uses
                   default KantoConfig values.
            rom_path: Path to Pokemon Red ROM file (.gb). If provided, overrides
                     the path in config. Either config.rom_path or this argument
                     must be provided.
            reward_fn: Reward function to use. Can be:
                      - String: 'default', 'badges_only', or 'exploration'
                      - RewardFunction instance: Custom reward function
            render_mode: How to render the game:
                        - 'human': Opens SDL2 window showing gameplay
                        - 'rgb_array': Returns screen as numpy array
                        - None: Headless mode (fastest for training)
            initial_step_offset: Number of steps to pre-advance the step counter
                   on the first episode only. This staggers episode truncation
                   across parallel environments so they don't all reset at the
                   same global step (which causes sawtooth metric patterns).

        Raises:
            ValueError: If no ROM path is provided via config or argument.

        Notes:
            - The emulator is not started until first step/reset (lazy init)
            - This allows creating many environments without memory overhead
        """
        super().__init__()

        # =================================================================
        # Configuration Setup
        # =================================================================

        # Use provided config or create default
        self.config = config or KantoConfig()

        # Override ROM path if explicitly provided
        if rom_path:
            self.config.rom_path = rom_path

        # Validate that we have a ROM path
        if not self.config.rom_path:
            raise ValueError(
                "ROM path must be provided via config or rom_path argument. "
                "Example: KantoRedEnv(rom_path='pokemon_red.gb')"
            )

        # =================================================================
        # Observation Producers
        # =================================================================

        # Build the configured observation components. This is the seam that
        # turns "pixels only" vs "full RAM observation" into a config choice.
        # The default tuple reproduces the historical observation exactly.
        self._producers = build_producers(self.config)
        self._want_screens = any(p.key == "screens" for p in self._producers)

        # =================================================================
        # Reward Function Setup
        # =================================================================

        # Create reward function from string name or use provided instance
        if isinstance(reward_fn, str):
            # Use factory function to create named reward function
            self.reward_fn = create_reward(
                reward_fn,
                reward_scale=self.config.reward_scale,
                explore_weight=self.config.explore_weight,
                interact_weight=self.config.interact_weight,
                stuck_penalty=self.config.stuck_penalty,
                level_weight=self.config.level_weight,
            )
        else:
            # Use provided reward function instance directly
            self.reward_fn = reward_fn

        # =================================================================
        # Rendering Setup
        # =================================================================

        # Store render mode for PyBoy window configuration
        self.render_mode = render_mode

        # =================================================================
        # Emulator State (Lazy Initialization)
        # =================================================================

        # PyBoy emulator instance - created on first access via property
        self._pyboy: PyBoy | None = None

        # Initial state bytes for fast reset
        # Stored after first initialization, used for all subsequent resets
        self._initial_state: bytes | None = None

        # =================================================================
        # Episode Tracking State
        # =================================================================

        # Step counter for current episode
        self.step_count = 0

        # Stagger offset: pre-advances step counter on first episode only
        # so parallel envs truncate at different times instead of all at once
        self._initial_step_offset = initial_step_offset
        self._first_episode = True

        # Previous game state for reward delta calculation
        self.prev_state: GameState | None = None

        # Recent action history for temporal awareness
        # Helps agent avoid repeating ineffective action patterns
        self.recent_actions = np.zeros(ACTION_HISTORY_LEN, dtype=np.int64)

        # =================================================================
        # Exploration Tracking
        # =================================================================

        # 48x48 bitmap tracking visited global coordinates
        # Each pixel represents a region of the Kanto map
        # Value 255 = visited, 0 = not visited
        self.explore_map = np.zeros(MAP_OBS_SHAPE, dtype=np.uint8)

        # Scale factors for converting global coords to exploration map coords
        # Kanto is ~444x436 tiles, we compress to 48x48
        self._explore_scale = (GLOBAL_MAP_WIDTH // 48, GLOBAL_MAP_HEIGHT // 48)

        # =================================================================
        # Event Flag Deduplication
        # =================================================================

        # Cached event count from _get_observation() — reused by reward and info
        # to avoid redundant 312-byte memory reads per step
        self._last_event_count: int = 0

        # =================================================================
        # Full Event Flag Tracking
        # =================================================================

        # Maps event flag bit indices to human-readable milestone names
        # Loaded once per process via _load_milestones() lazy cache
        self._milestone_map: dict[int, str] = _load_milestones()

        # Previous step's event flags — used for step-to-step diff
        # Starts as zeros so first observation captures all pre-set events
        self._prev_events: np.ndarray = np.zeros(EVENT_OBS_DIM, dtype=np.int8)

        # Bit indices of events newly set THIS step (for IPC to callbacks)
        self._new_event_indices: list[int] = []

        # =================================================================
        # Frame Stacking
        # =================================================================

        # Stack of recent frames for temporal information
        # Shape: (frame_stacks, height, width) e.g., (3, 72, 80)
        self.frame_stack = np.zeros(
            (self.config.frame_stacks, *self.config.screen_size),
            dtype=np.uint8,
        )

        # =================================================================
        # Define Gymnasium Spaces
        # =================================================================

        self._define_spaces()

    def _define_spaces(self) -> None:
        """
        Derive the Gymnasium spaces from the active observation producers.

        The observation space is a Dict whose keys and subspaces come from
        ``KantoConfig.obs_components``. Because each producer owns both its
        ``space()`` and its ``read()``, the declared space and the actual
        observation cannot drift apart.

        The action space remains Discrete(8):
        0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START.
        """
        self.observation_space = spaces.Dict(
            {producer.key: producer.space() for producer in self._producers}
        )

        # Define discrete action space
        # 8 actions: NOOP, UP, DOWN, LEFT, RIGHT, A, B, START
        self.action_space = spaces.Discrete(8)

    @property
    def pyboy(self) -> PyBoy:
        """
        Lazy initialization of the PyBoy emulator.

        The emulator is only created when first accessed, allowing many
        environment instances to be created without immediate memory cost.
        This is important for vectorized training with many parallel envs.

        Returns:
            Initialized PyBoy emulator instance.

        Notes:
            - Creates SDL2 window if render_mode='human', else headless
            - Loads save state for consistent starting point
            - Falls back to intro-skipping if no save state found
            - Caches initial state for fast reset
        """
        if self._pyboy is None:
            # =============================================================
            # Create PyBoy Instance
            # =============================================================

            # Choose window mode based on render setting
            # "null" = headless (fast), "SDL2" = visual window
            window = "null" if self.render_mode != "human" else "SDL2"

            # Create the emulator with the ROM
            # sound=False skips audio emulation — wasted CPU in headless training
            self._pyboy = PyBoy(
                self.config.rom_path,
                window=window,
                sound=False,
            )

            # =============================================================
            # Load Initial Game State
            # =============================================================

            # Check for save state file
            state_path = self.config.save_state_path

            if not state_path:
                # If no explicit path, look for common state files next to ROM
                rom_dir = Path(self.config.rom_path).parent

                # Try known state file names in order of preference
                for name in ["has_pokedex_nballs.state", "init.state"]:
                    candidate = rom_dir / name
                    if candidate.exists():
                        state_path = str(candidate)
                        break

            # Load state if found, otherwise skip intro manually
            if state_path and Path(state_path).exists():
                # Load save state for reliable starting point
                with open(state_path, "rb") as f:
                    self._pyboy.load_state(f)
            else:
                # Fallback: skip intro by pressing buttons
                # Less reliable but works without save state
                self._skip_intro()

            # =============================================================
            # Cache Initial State for Fast Reset
            # =============================================================

            # Save current state to bytes for fast reloading
            # This avoids re-reading the state file on every reset
            state_buffer = BytesIO()
            self._pyboy.save_state(state_buffer)
            self._initial_state = state_buffer.getvalue()

        return self._pyboy

    def _skip_intro(self) -> None:
        """
        Skip the ROM intro sequence by pressing buttons.

        This is a fallback method when no save state is available.
        It advances through the intro screens by pressing A repeatedly.

        Notes:
            - Less reliable than loading a save state
            - May leave the game in slightly different states
            - Takes ~5 seconds of emulated time
        """
        # Run emulator for a few seconds to get past initial logo
        for _ in range(300):
            self._pyboy.tick()

        # Press A repeatedly to skip through intro text
        for _ in range(5):
            self._pyboy.button("a")
            # Wait 60 frames (~1 second) between presses
            for _ in range(60):
                self._pyboy.tick()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to the initial state.

        Reloads the game from the cached initial state and resets all
        tracking variables. This is called at the start of each episode.

        Args:
            seed: Random seed for reproducibility. Currently unused as the
                 environment is deterministic, but accepted for API compliance.
            options: Additional reset options. Currently unused but accepted
                    for API compliance.

        Returns:
            Tuple of (observation, info):
            - observation: Dict with all observation components
            - info: Dict with game state information for logging

        Notes:
            - Uses cached state bytes for fast reset (~instant)
            - Resets exploration map, frame stack, and action history
            - Calls reward function reset for fresh episode tracking
        """
        # Call parent reset for seed handling
        super().reset(seed=seed)

        # =================================================================
        # Reset Emulator State
        # =================================================================

        if self._initial_state is not None:
            # Fast path: load from cached state bytes
            state_buffer = BytesIO(self._initial_state)
            self.pyboy.load_state(state_buffer)
        else:
            # First reset: trigger lazy initialization
            _ = self.pyboy

        # =================================================================
        # Reset Episode Tracking
        # =================================================================

        # On the first episode, apply the stagger offset so parallel envs
        # truncate at different times instead of all at once.
        # After the first episode, envs are naturally desynchronized.
        if self._first_episode and self._initial_step_offset > 0:
            self.step_count = self._initial_step_offset
            self._first_episode = False
        else:
            self.step_count = 0

        # Clear previous state (no delta on first step)
        self.prev_state = None

        # Clear action history
        self.recent_actions = np.zeros(ACTION_HISTORY_LEN, dtype=np.int64)

        # Clear exploration map
        self.explore_map = np.zeros(MAP_OBS_SHAPE, dtype=np.uint8)

        # Clear frame stack
        self.frame_stack = np.zeros_like(self.frame_stack)

        # Clear any producer-local caches (e.g. level encoding)
        for producer in self._producers:
            producer.reset()

        # Reset event flag tracking for new episode
        self._prev_events = np.zeros(EVENT_OBS_DIM, dtype=np.int8)
        self._new_event_indices = []

        # Reset reward function internal state
        self.reward_fn.reset()

        # =================================================================
        # Build Initial Observation and Info
        # =================================================================

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Execute one action in the environment.

        Processes the action by pressing the corresponding button, advances
        the emulator, calculates the reward, and builds the new observation.

        Args:
            action: Action index (0-7) corresponding to:
                   0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=A, 6=B, 7=START

        Returns:
            Tuple of (observation, reward, terminated, truncated, info):
            - observation: Dict with all observation components
            - reward: Float reward for this step
            - terminated: Always False (Pokemon Red has no natural ending)
            - truncated: True if max_steps reached
            - info: Dict with game state information

        Notes:
            - Action execution uses frame skipping (action_freq frames)
            - Reward is calculated from state delta
            - Episode ends by truncation at max_steps, not termination
        """
        # =================================================================
        # Execute Action
        # =================================================================

        # Press button and advance emulator
        self._execute_action(action)

        # Increment step counter
        self.step_count += 1

        # =================================================================
        # Update Action History
        # =================================================================

        # Shift action history left and add new action
        # np.roll with -1 moves elements left, oldest falls off
        self.recent_actions = np.roll(self.recent_actions, -1)
        self.recent_actions[-1] = action

        # =================================================================
        # Update Exploration Map
        # =================================================================

        self._update_explore_map()

        # =================================================================
        # Build Observation (reads event flags once)
        # =================================================================

        obs = self._get_observation()

        # =================================================================
        # Calculate Reward (reuses event count from observation)
        # =================================================================

        # Pass pre-computed event count to avoid redundant 312-byte memory read
        current_state = GameState.from_pyboy(
            self.pyboy, self.step_count, event_count=self._last_event_count
        )

        # Calculate reward from state change
        reward = self.reward_fn.calculate(current_state, self.prev_state, action)

        # Store state for next step's delta
        self.prev_state = current_state

        # =================================================================
        # Check Episode Termination
        # =================================================================

        # Pokemon Red doesn't have natural termination (no "game over")
        # Even fainting just sends you to a Pokemon Center
        terminated = False

        # Truncate episode at max_steps to bound episode length
        truncated = self.step_count >= self.config.max_steps

        # =================================================================
        # Build Info (reuses event count from observation)
        # =================================================================

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> None:
        """
        Execute an action by pressing the button and advancing frames.

        Translates the discrete action index to button press/release events
        and advances the emulator by action_freq frames.

        Args:
            action: Action index (0-7).

        Notes:
            - Button is held for 8 frames before release
            - Total frame advance equals config.action_freq
            - NOOP (action 0) just advances frames without button press
        """
        # Look up button events for this action
        press = PRESS_ACTIONS[action]
        release = RELEASE_ACTIONS[action]

        # Press the button (if not NOOP)
        if press:
            self.pyboy.send_input(press)

        # Hold button for 8 frames (reliable input registration)
        press_frames = 8
        self.pyboy.tick(press_frames, False)  # False = don't render

        # Release the button
        if release:
            self.pyboy.send_input(release)

        # Advance remaining frames to reach action_freq total
        remaining = self.config.action_freq - press_frames
        if remaining > 0:
            self.pyboy.tick(remaining, False)

    def _get_observation(self) -> dict[str, np.ndarray]:
        """
        Build the observation dict from the active producers.

        Memory is read once per step and placed in an ObsContext; producers
        then transform only the fields they need. This preserves the original
        single-read behavior (event flags are still unpacked once and reused
        by reward/info) while allowing components to be dropped from the tier.

        Returns:
            Dict whose keys match ``self.observation_space.spaces``.
        """
        # Update the frame stack only if the screen is actually observed.
        # RAM-only tiers can skip the RGB conversion and resize entirely.
        if self._want_screens:
            screen = self._get_screen_gray()
            self.frame_stack[:-1] = self.frame_stack[1:]
            self.frame_stack[-1] = screen

        # Read shared state once per step.
        current_hp, max_hp = memory.get_total_party_hp(self.pyboy)

        levels: tuple[int, ...] = ()
        if "level" in self.config.obs_components:
            levels = tuple(memory.get_party_levels(self.pyboy))

        badge_flags = None
        if "badges" in self.config.obs_components:
            badge_flags = memory.get_badge_flags(self.pyboy)

        # Event flags are needed by the reward and info dict even when they
        # are not exposed to the policy, so this read is unconditional.
        events = memory.get_event_flags(self.pyboy)
        self._last_event_count = int(np.sum(events))

        # Vectorized diff finds every flag that flipped 0→1 this step.
        newly_set = np.where((events == 1) & (self._prev_events == 0))[0]
        self._new_event_indices = newly_set.tolist()
        self._prev_events = events.copy()

        ctx = ObsContext(
            pyboy=self.pyboy,
            frame_stack=self.frame_stack if self._want_screens else None,
            explore_map=self.explore_map,
            recent_actions=self.recent_actions,
            event_flags=events,
            badge_flags=badge_flags,
            levels=levels,
            current_hp=current_hp,
            max_hp=max_hp,
        )

        # No .copy() needed: SubprocVecEnv pickles observations for IPC
        # (inherent copy), and originals are overwritten on the next step.
        return {producer.key: producer.read(ctx) for producer in self._producers}

    def _get_screen_gray(self) -> np.ndarray:
        """
        Get the current screen as a downscaled grayscale image.

        Converts the PyBoy screen (144x160 RGB) to grayscale and resizes
        to the configured screen_size for efficient processing.

        Returns:
            Grayscale screen as uint8 array with shape (height, width).

        Notes:
            - Original GameBoy resolution: 144x160 (height x width)
            - Default target: 72x80 (4x reduction)
            - Uses OpenCV INTER_AREA for fast, anti-aliased downscaling
            - PyBoy screen is RGBA (4 channels); we slice to RGB before converting
        """
        # Get RGBA screen from PyBoy (shape: 144, 160, 4) and slice to RGB
        screen = self.pyboy.screen.ndarray[:, :, :3]

        # Convert RGB to grayscale in C (no Python-level type juggling)
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        # Resize to target dimensions using INTER_AREA (optimal for downscaling)
        # cv2.resize takes (width, height) — opposite of numpy's (height, width)
        h, w = self.config.screen_size
        return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

    def _update_explore_map(self) -> None:
        """
        Update the exploration bitmap with the current position.

        Converts the current local map coordinates to global Kanto
        coordinates, then marks the corresponding pixel in the
        exploration map.

        Notes:
            - Global coordinates cover the entire Kanto region
            - Exploration map is 48x48, downscaled from ~444x436
            - Each exploration map pixel represents ~9x9 game tiles
            - Value 255 marks visited, 0 marks unvisited
        """
        # Get current position (map_id, local_x, local_y)
        map_id, x, y = memory.get_position(self.pyboy)

        # Convert to global Kanto coordinates
        global_x, global_y = local_to_global(map_id, x, y)

        # Scale down to exploration map coordinates
        # Clamp to valid range [0, 47] to avoid index errors
        map_x = min(global_x // self._explore_scale[0], 47)
        map_y = min(global_y // self._explore_scale[1], 47)

        # Mark as visited (255 = max uint8 value)
        self.explore_map[map_y, map_x] = 255

    def _get_info(self) -> dict[str, Any]:
        """
        Build the info dictionary for logging and debugging.

        Contains game state information that isn't part of the observation
        but is useful for monitoring training progress.

        Returns:
            Dict with game state info:
            - step: Current step count
            - map_id: Current map ID
            - position: (x, y) position tuple
            - badges: Number of badges
            - events: Number of event flags set
            - party_count: Number of Pokemon in party
            - in_battle: Whether currently in battle
            - explore_tiles: Number of exploration map tiles visited
            - Plus all keys from reward_fn.get_info()
        """
        # Get current position
        map_id, x, y = memory.get_position(self.pyboy)

        # Get reward function info (breakdown, stats, etc.)
        reward_info = self.reward_fn.get_info()

        # Build info dict
        return {
            "step": self.step_count,
            "map_id": map_id,
            "position": (x, y),
            "badges": memory.get_badges(self.pyboy),
            "events": self._last_event_count,
            "new_event_indices": self._new_event_indices,
            "party_count": memory.get_party_count(self.pyboy),
            "in_battle": memory.is_in_battle(self.pyboy),
            "explore_tiles": int(np.sum(self.explore_map > 0)),
            **reward_info,  # Merge reward function info
        }

    def render(self) -> np.ndarray | None:
        """
        Render the current frame.

        Returns the screen as an RGB array if render_mode='rgb_array'.
        For render_mode='human', the SDL2 window handles display automatically.

        Returns:
            RGB array of shape (144, 160, 3) if render_mode='rgb_array'.
            None for other render modes.
        """
        if self.render_mode == "rgb_array":
            # Return copy of screen array
            return self.pyboy.screen.ndarray.copy()
        elif self.render_mode == "human":
            # SDL2 window updates automatically with emulator ticks
            pass
        return None

    def close(self) -> None:
        """
        Clean up resources and close the emulator.

        Should be called when done with the environment to free memory
        and close any open windows.
        """
        if self._pyboy is not None:
            self._pyboy.stop()
            self._pyboy = None


# =============================================================================
# VECTORIZED ENVIRONMENT FACTORY
# =============================================================================


def make_env(
    rom_path: str,
    config: KantoConfig | None = None,
    rank: int = 0,
    seed: int = 0,
    reward_fn: str = "default",
    enable_streaming: bool = False,
    enable_curriculum: bool = False,
    enable_agent: bool = False,
    initial_step_offset: int = 0,
) -> callable:
    """
    Factory function for creating environments in vectorized setup.

    Returns a callable that creates and initializes an environment.
    This is the format required by SubprocVecEnv and DummyVecEnv.

    Args:
        rom_path: Path to Pokemon Red ROM file.
        config: Configuration object. If None, uses defaults.
        rank: Environment rank/index for unique seeding.
        seed: Base random seed (actual seed = seed + rank).
        reward_fn: Reward function name ('default', 'badges_only', 'exploration').
        enable_streaming: Whether to wrap with StreamWrapper for visualization.
        enable_curriculum: Whether to wrap with CurriculumWrapper for curriculum learning.
        enable_agent: Whether to wrap with AgentWrapper for modular agent system.
        initial_step_offset: Step counter offset for the first episode, used to
            stagger episode truncation across parallel environments. Routed to
            CurriculumWrapper if curriculum is active, otherwise to KantoRedEnv.

    Returns:
        Callable that returns an initialized Gymnasium environment.

    Example:
        >>> # Create 4 parallel environments
        >>> env_fns = [make_env("pokemon_red.gb", rank=i) for i in range(4)]
        >>> vec_env = SubprocVecEnv(env_fns)

    Notes:
        - Each environment gets a unique seed based on rank
        - StreamWrapper is optionally applied for real-time visualization
        - AgentWrapper is applied after CurriculumWrapper, before StreamWrapper
        - The returned callable is invoked by the vectorized env wrapper
    """

    def _init() -> gym.Env:
        """
        Initialize and return the environment.

        This inner function is called by SubprocVecEnv to create each
        environment in its own process.
        """
        # Create or use provided config
        cfg = config or KantoConfig(rom_path=rom_path)
        cfg.rom_path = rom_path

        # Create the base environment
        # If curriculum is active, the stagger offset is handled by CurriculumWrapper
        # (which overrides truncation logic). Otherwise, KantoRedEnv handles it.
        env = KantoRedEnv(
            config=cfg,
            reward_fn=reward_fn,
            initial_step_offset=initial_step_offset if not enable_curriculum else 0,
        )

        # Reset with unique seed based on rank
        env.reset(seed=seed + rank)

        # Optionally wrap with CurriculumWrapper for curriculum learning
        # Must come before StreamWrapper since it modifies game state
        if enable_curriculum and cfg.enable_curriculum:
            from kantorl.curriculum import CurriculumWrapper

            env = CurriculumWrapper(
                env,
                pool_dir=cfg.session_path / "checkpoint_pool",
                config=cfg,
                initial_step_offset=initial_step_offset,
            )

        # Optionally wrap with AgentWrapper for modular agent system
        # Must come after CurriculumWrapper (needs post-checkpoint game state)
        # and before StreamWrapper (streaming doesn't need agent observations)
        if enable_agent and cfg.enable_agent:
            from kantorl.agent import AgentWrapper

            env = AgentWrapper(env, config=cfg)

        # Optionally wrap with StreamWrapper for visualization
        if enable_streaming and cfg.enable_streaming:
            from kantorl.stream_wrapper import StreamWrapper

            # Create unique username for each parallel environment
            # Always append rank for consistent naming (e.g., "KantoRL_1", "KantoRL_2")
            username = f"{cfg.stream_username}_{rank + 1}"

            # Wrap environment with streaming capability
            # Pass rank and n_envs so StreamWrapper can auto-generate a unique
            # color per agent when no explicit per-agent color is configured
            env = StreamWrapper(
                env,
                username=username,
                color=cfg.stream_color,
                sprite_id=cfg.stream_sprite_id,
                stream_interval=cfg.stream_interval,
                extra_info=cfg.stream_extra,
                enabled=True,
                rank=rank,
                n_envs=cfg.n_envs,
            )

        return env

    return _init
