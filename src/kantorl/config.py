"""
Configuration dataclass for KantoRL.

This module provides the central configuration system for the KantoRL project.
All hyperparameters, paths, and settings are defined in a single KantoConfig
dataclass, making it easy to understand, modify, and validate the configuration.

Key Features:
    - Type-safe configuration using Python dataclasses
    - Automatic validation of parameters in __post_init__
    - Sensible defaults matching PokemonRedExperiments V2
    - Easy serialization via from_dict() class method

Architecture Role:
    KantoConfig is used by:
    - env.py: Environment creation and observation settings
    - train.py: Training hyperparameters and session paths
    - callbacks.py: Checkpoint and logging configuration
    - stream_wrapper.py: Streaming settings for visualization

Example Usage:
    >>> config = KantoConfig(rom_path="pokemon_red.gb", n_envs=8)
    >>> config.learning_rate
    0.0003
    >>> config = KantoConfig.from_dict({"rom_path": "game.gb", "n_envs": 4})

Dependencies:
    - dataclasses: For the dataclass decorator and field function
    - pathlib: For Path objects used in file path handling
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KantoConfig:
    """
    Configuration for KantoRL environment and training.

    This dataclass contains all configurable parameters for the Pokemon Red
    reinforcement learning environment. Parameters are organized into logical
    groups: paths, emulator settings, observation settings, reward settings,
    training hyperparameters, and streaming settings.

    Attributes:
        rom_path (str): Path to Pokemon Red ROM file (.gb). Required for training.
        save_state_path (str | None): Optional path to .state file for starting point.
            If None, looks for 'has_pokedex_nballs.state' or 'init.state' next to ROM.
        session_path (Path): Directory for saving checkpoints and TensorBoard logs.
            Defaults to "runs" in the current working directory.

        headless (bool): Run without display window. True for training (faster),
            False for visualization/debugging.
        action_freq (int): Number of emulator frames between agent actions (frame skip).
            Higher values = faster training but less precise control.
            24 frames ≈ 0.4 seconds of game time at 60 FPS.
        max_steps (int): Maximum steps per episode before truncation.
            Default: 2048 * 80 = 163,840 steps ≈ 1.8 hours of game time.

        frame_stacks (int): Number of consecutive frames to stack for temporal info.
            Helps the agent perceive motion and changes over time.
        screen_size (tuple[int, int]): Downscaled screen dimensions (height, width).
            Original GameBoy: 144x160. Default 72x80 = 4x reduction.

        reward_scale (float): Global multiplier for all rewards.
            Lower values encourage exploration, higher values exploit known strategies.
        explore_weight (float): Weight for exploration bonus in reward function.
            Higher values encourage visiting new map coordinates.

        n_envs (int): Number of parallel environments for vectorized training.
            More envs = more diverse experience but higher memory usage.
        n_steps (int): Steps collected per environment before each PPO update.
            Total batch = n_envs * n_steps = 16 * 64 = 1024 steps.
        batch_size (int): Minibatch size for PPO gradient updates.
            Should divide evenly into n_envs * n_steps.

        learning_rate (float): Adam optimizer learning rate for PPO.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): GAE (Generalized Advantage Estimation) lambda.
        clip_range (float): PPO clipping parameter for policy updates.
        ent_coef (float): Entropy coefficient for exploration.
        vf_coef (float): Value function loss coefficient.
        n_epochs (int): Number of epochs per PPO update.

        enable_streaming (bool): Enable WebSocket streaming to shared map visualization.
        stream_username (str): Display name for this agent on the shared map.
        stream_color (str): Hex color code for map marker (e.g., "#0033ff").
        stream_sprite_id (int): Character sprite ID (0-50) for map display.
        stream_interval (int): Steps between coordinate uploads to server.
        stream_extra (str): Additional text to display on the map.

    Example:
        >>> config = KantoConfig(
        ...     rom_path="pokemon_red.gb",
        ...     n_envs=8,
        ...     learning_rate=1e-4,
        ... )
        >>> print(f"Training with {config.n_envs} environments")
        Training with 8 environments

    Notes:
        - All PPO hyperparameters are tuned to match PokemonRedExperiments V2
        - The config is validated in __post_init__ to catch invalid values early
        - Use from_dict() to create configs from JSON/YAML files
    """

    # =============================================================================
    # PATH CONFIGURATION
    # =============================================================================

    # Path to the Pokemon Red ROM file (.gb format)
    # This is the only truly required parameter - everything else has defaults
    rom_path: str = ""

    # Optional path to a save state file (.state format from PyBoy)
    # Save states allow starting from a specific point in the game
    # (e.g., after character creation, with Pokedex and Pokeballs)
    save_state_path: str | None = None

    # Directory for training artifacts (checkpoints, TensorBoard logs)
    # Created automatically if it doesn't exist
    session_path: Path = field(default_factory=lambda: Path("runs"))

    # =============================================================================
    # EMULATOR SETTINGS
    # =============================================================================

    # Run PyBoy without a display window
    # True: Faster training, no visual output (uses "null" window)
    # False: Shows game window via SDL2 (useful for debugging/demo)
    headless: bool = True

    # Frame skip: number of emulator frames between agent decisions
    # GameBoy runs at 60 FPS, so action_freq=24 means ~2.5 actions per second
    # Higher values = faster training but coarser control
    # Lower values = finer control but slower training and harder credit assignment
    # 24 is a good balance for Pokemon Red's turn-based nature
    action_freq: int = 24

    # Maximum steps per episode before forced termination (truncation)
    # Default: 2048 * 80 = 163,840 steps
    # At action_freq=24 and 60 FPS: ~109 minutes of game time per episode
    # This is long enough to make significant progress but prevents infinite episodes
    max_steps: int = 2048 * 80

    # =============================================================================
    # OBSERVATION SETTINGS
    # =============================================================================

    # Number of consecutive frames to stack in the observation
    # Frame stacking provides temporal information (motion, changes)
    # 3 frames is standard for Atari-style environments
    frame_stacks: int = 3

    # Target screen size after downscaling (height, width)
    # Original GameBoy resolution: 144x160 pixels
    # 72x80 is a 4x reduction, balancing detail vs computational cost
    # Must be a tuple of (height, width) to match numpy array conventions
    screen_size: tuple[int, int] = (72, 80)

    # =============================================================================
    # REWARD SETTINGS
    # =============================================================================

    # Global multiplier applied to all reward components
    # Affects the magnitude of gradients during training
    # 0.5 provides moderate reward signals without overwhelming the value function
    reward_scale: float = 0.5

    # Weight for the exploration bonus component of the reward
    # Higher values encourage visiting new map coordinates
    # 0.1 provides gentle exploration pressure without dominating other rewards
    explore_weight: float = 0.1

    # =============================================================================
    # TRAINING HYPERPARAMETERS (PPO)
    # These values are tuned to match PokemonRedExperiments V2 for reproducibility
    # =============================================================================

    # Number of parallel environments for vectorized training
    # More environments = more diverse experience per update
    # 16 is a good balance of parallelism vs memory usage
    n_envs: int = 16

    # Steps to collect per environment before each PPO update
    # Total experiences per update = n_envs * n_steps = 16 * 64 = 1024
    n_steps: int = 64

    # Minibatch size for PPO gradient updates
    # Should divide evenly into (n_envs * n_steps)
    # Smaller batches = noisier gradients but more updates per epoch
    batch_size: int = 256

    # Learning rate for the Adam optimizer
    # 3e-4 is the standard PPO learning rate from the original paper
    learning_rate: float = 3e-4

    # Discount factor (gamma) for future rewards
    # 0.997 means rewards far in the future are still somewhat valuable
    # Higher than Atari (0.99) because Pokemon has longer-term goals
    gamma: float = 0.997

    # GAE (Generalized Advantage Estimation) lambda parameter
    # Balances bias (low lambda) vs variance (high lambda) in advantage estimates
    # 0.95 is the standard value from PPO paper
    gae_lambda: float = 0.95

    # PPO clipping parameter
    # Limits the ratio of new/old policy probabilities
    # 0.1 is slightly more conservative than the default 0.2
    # Smaller values = more stable but slower learning
    clip_range: float = 0.1

    # Entropy coefficient for the loss function
    # Adds bonus for high-entropy (random) action distributions
    # Encourages exploration and prevents premature convergence
    ent_coef: float = 0.01

    # Value function coefficient for the loss function
    # Balances policy loss vs value function loss
    # 0.5 gives equal weight to both objectives
    vf_coef: float = 0.5

    # Number of optimization epochs per PPO update
    # More epochs = more thorough optimization of collected experience
    # But too many can cause overfitting to the current batch
    n_epochs: int = 4

    # =============================================================================
    # STREAMING SETTINGS
    # Settings for broadcasting game coordinates to a shared visualization server
    # This allows multiple training sessions to be visualized on a single map
    # =============================================================================

    # Master switch for streaming functionality
    # When False, StreamWrapper is not applied even if configured
    enable_streaming: bool = False

    # Display name for this agent on the shared map
    # Should be unique to identify different training runs
    stream_username: str = "kantorl-agent"

    # Hex color code for this agent's marker on the map
    # Format: "#RRGGBB" (e.g., "#0033ff" for blue)
    stream_color: str = "#0033ff"

    # Character sprite ID for map visualization (0-50)
    # Different IDs show different Pokemon trainer sprites
    stream_sprite_id: int = 0

    # Number of steps between coordinate uploads
    # Lower values = more responsive visualization but more network traffic
    # 300 steps ≈ 5 seconds at typical training speed
    stream_interval: int = 300

    # Additional text to display alongside this agent's marker
    # Can be used to show training progress, hyperparameters, etc.
    stream_extra: str = ""

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        This method is automatically called by the dataclass after __init__.
        It performs validation checks on the configuration values and converts
        types where necessary.

        Raises:
            ValueError: If action_freq < 1 (must have at least 1 frame per action)
            ValueError: If frame_stacks < 1 (must have at least 1 frame)
            ValueError: If reward_scale not in (0, 10] (prevents zero/negative scaling)

        Notes:
            - session_path is converted from str to Path if necessary
            - Additional validations can be added here as needed
        """
        # Convert session_path to Path object if it's a string
        # This allows users to pass either str or Path when creating config
        if isinstance(self.session_path, str):
            self.session_path = Path(self.session_path)

        # Validate action_freq: must be positive to avoid division by zero
        # and ensure at least one frame passes per action
        if self.action_freq < 1:
            raise ValueError(
                f"action_freq must be >= 1, got {self.action_freq}. "
                "The agent needs at least 1 frame to observe the result of an action."
            )

        # Validate frame_stacks: must have at least one frame in observation
        if self.frame_stacks < 1:
            raise ValueError(
                f"frame_stacks must be >= 1, got {self.frame_stacks}. "
                "The observation must include at least one frame."
            )

        # Validate reward_scale: must be positive and reasonable
        # Upper bound of 10 prevents accidentally huge rewards that destabilize training
        if not (0 < self.reward_scale <= 10):
            raise ValueError(
                f"reward_scale should be in (0, 10], got {self.reward_scale}. "
                "Zero or negative scaling would eliminate or invert rewards."
            )

    @classmethod
    def from_dict(cls, d: dict) -> "KantoConfig":
        """
        Create a KantoConfig from a dictionary, ignoring unknown keys.

        This factory method allows creating configs from JSON files, YAML files,
        or any dictionary source while gracefully ignoring extra keys that aren't
        part of the config schema.

        Args:
            d: Dictionary containing configuration values. Unknown keys are ignored.

        Returns:
            A new KantoConfig instance with values from the dictionary.
            Missing keys use their default values.

        Example:
            >>> config_dict = {
            ...     "rom_path": "pokemon_red.gb",
            ...     "n_envs": 8,
            ...     "unknown_key": "ignored"  # This won't cause an error
            ... }
            >>> config = KantoConfig.from_dict(config_dict)
            >>> config.n_envs
            8

        Notes:
            - Only keys that match dataclass field names are used
            - This allows forward/backward compatibility with config files
            - Type conversion is not performed; values should be correct types
        """
        # Get the set of valid field names from the dataclass definition
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}

        # Filter the input dictionary to only include valid keys
        # This prevents TypeError from unexpected keyword arguments
        filtered = {k: v for k, v in d.items() if k in valid_keys}

        # Create and return the config with filtered values
        return cls(**filtered)
