"""
Training callbacks for KantoRL.

This module provides stable-baselines3 compatible callbacks for monitoring,
checkpointing, and analyzing training progress. Callbacks are invoked at
each training step and can log metrics, save models, or detect issues.

Architecture Role:
    Callbacks are passed to PPO.learn() in train.py and execute during training.
    They provide the following functionality:
    - CumulativeCheckpointCallback: Save models with cumulative step counts
    - TensorboardCallback: Log game-specific metrics (badges, events, exploration)
    - StallDetectionCallback: Warn when training progress stalls
    - PerformanceCallback: Monitor training speed (steps/second)

Key Concepts:
    - BaseCallback: All callbacks inherit from stable-baselines3's BaseCallback
    - _on_step(): Called after each environment step, return True to continue
    - self.num_timesteps: Total steps across all environments
    - self.model: Reference to the PPO model being trained
    - self.logger: TensorBoard logger for recording metrics
    - self.locals: Dictionary of local variables from the training loop

Usage:
    callbacks = CallbackList([
        CumulativeCheckpointCallback(save_path="checkpoints", save_freq=100_000),
        TensorboardCallback(log_freq=1000),
        StallDetectionCallback(check_freq=50_000),
        PerformanceCallback(log_freq=10_000),
    ])
    model.learn(total_timesteps=1_000_000, callback=callbacks)

Dependencies:
    - stable_baselines3: For BaseCallback base class
    - numpy: For metric aggregation
    - time: For performance timing
    - pathlib: For checkpoint path handling
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# =============================================================================
# CHECKPOINT CALLBACK
# =============================================================================


class CumulativeCheckpointCallback(BaseCallback):
    """
    Save model checkpoints with cumulative step counts in the filename.

    Unlike standard checkpoints that overwrite a single file, this callback
    creates uniquely named checkpoints (e.g., model_100000.zip, model_200000.zip)
    so you never lose previous checkpoints. This is essential for:
    - Comparing models at different training stages
    - Recovering from training issues
    - Analyzing learning curves

    The callback also provides a class method to find the latest checkpoint
    for auto-resume functionality.

    Attributes:
        save_path (Path): Directory where checkpoints are saved.
        save_freq (int): Save a checkpoint every N steps.
        name_prefix (str): Prefix for checkpoint filenames.
        verbose (int): Verbosity level (0=silent, 1=print saves).

    Example:
        >>> callback = CumulativeCheckpointCallback(
        ...     save_path="runs/checkpoints",
        ...     save_freq=100_000,
        ... )
        >>> # After training, checkpoints will be:
        >>> # runs/checkpoints/model_100000.zip
        >>> # runs/checkpoints/model_200000.zip
        >>> # etc.

    Notes:
        - Checkpoints are saved as .zip files containing model weights and config
        - The save_path directory is created if it doesn't exist
        - Step counts are cumulative across resumed training sessions
    """

    def __init__(
        self,
        save_path: Path | str,
        save_freq: int = 100_000,
        name_prefix: str = "model",
        verbose: int = 1,
    ):
        """
        Initialize the checkpoint callback.

        Args:
            save_path: Directory to save checkpoints. Created if it doesn't exist.
            save_freq: Save a checkpoint every N training steps.
                       100,000 is a good default (saves ~10x per million steps).
            name_prefix: Prefix for checkpoint filenames. Default "model" produces
                        files like "model_100000.zip".
            verbose: Verbosity level. 0=silent, 1=print when checkpoints are saved.
        """
        super().__init__(verbose)

        # Convert string path to Path object for consistent handling
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.name_prefix = name_prefix

        # Create the checkpoint directory if it doesn't exist
        # parents=True creates intermediate directories, exist_ok=True ignores if exists
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called after each training step. Saves checkpoint if at save_freq interval.

        This method is called by the training loop after each step. It checks
        if the current step count is a multiple of save_freq and saves a
        checkpoint if so.

        Returns:
            True to continue training, False to stop.
            Always returns True (checkpointing never stops training).
        """
        # Check if we're at a save interval
        # num_timesteps is the total steps across all environments
        if self.num_timesteps % self.save_freq == 0:
            # Create checkpoint filename with step count
            path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}.zip"

            # Save the model (includes policy network, optimizer state, etc.)
            self.model.save(path)

            # Print save message if verbose
            if self.verbose:
                print(f"Saved checkpoint: {path}")

        # Always continue training
        return True

    @classmethod
    def find_latest(
        cls, save_path: Path | str, name_prefix: str = "model"
    ) -> tuple[Path | None, int]:
        """
        Find the latest checkpoint and its step count for auto-resume.

        Scans the checkpoint directory for files matching the naming pattern
        and returns the one with the highest step count.

        Args:
            save_path: Directory containing checkpoints.
            name_prefix: Checkpoint file prefix to search for.

        Returns:
            Tuple of (checkpoint_path, step_count).
            Returns (None, 0) if no checkpoints are found.

        Example:
            >>> path, steps = CumulativeCheckpointCallback.find_latest("runs/checkpoints")
            >>> if path:
            ...     print(f"Found checkpoint at {steps:,} steps: {path}")
            ...     model = PPO.load(path)
            ... else:
            ...     print("No checkpoint found, starting fresh")

        Notes:
            - Only looks for .zip files matching the pattern {prefix}_*.zip
            - Step count is extracted from the filename (e.g., model_100000.zip -> 100000)
            - Invalid filenames (non-numeric suffix) are ignored
        """
        save_path = Path(save_path)

        # Check if directory exists
        if not save_path.exists():
            return None, 0

        # Find all checkpoint files matching the pattern
        checkpoints = list(save_path.glob(f"{name_prefix}_*.zip"))
        if not checkpoints:
            return None, 0

        # Helper function to extract step count from filename
        def get_steps(p: Path) -> int:
            """Extract step count from checkpoint filename."""
            try:
                # Filename format: model_100000.zip -> stem is model_100000
                # Split by underscore and take the last part as the step count
                return int(p.stem.split("_")[-1])
            except ValueError:
                # If parsing fails, return 0 (will be sorted to the bottom)
                return 0

        # Find the checkpoint with the highest step count
        latest = max(checkpoints, key=get_steps)
        return latest, get_steps(latest)


# =============================================================================
# TENSORBOARD LOGGING CALLBACK
# =============================================================================


class TensorboardCallback(BaseCallback):
    """
    Log game-specific metrics to TensorBoard.

    While stable-baselines3 automatically logs standard RL metrics (reward,
    episode length, loss), this callback logs Pokemon Red specific metrics:
    - Badges collected (key progress milestone)
    - Event flags triggered (fine-grained progress)
    - Unique coordinates visited (exploration)

    These metrics help diagnose whether the agent is actually learning to
    play the game, not just optimizing reward.

    Attributes:
        log_freq (int): Log aggregated metrics every N steps.
        badges_collected (list): Buffer for badge counts from each environment.
        events_triggered (list): Buffer for event counts.
        unique_coords (list): Buffer for exploration tile counts.

    Example:
        >>> callback = TensorboardCallback(log_freq=1000)
        >>> # View in TensorBoard:
        >>> # tensorboard --logdir runs/tensorboard

    Notes:
        - Metrics are aggregated across all parallel environments
        - Buffers are cleared after each log to track recent performance
        - Metrics appear under the "game/" prefix in TensorBoard
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        """
        Initialize the TensorBoard callback.

        Args:
            log_freq: Log aggregated metrics every N training steps.
                     1000 steps provides good granularity without too much data.
            verbose: Verbosity level (currently unused, for API consistency).
        """
        super().__init__(verbose)
        self.log_freq = log_freq

        # Buffers to collect metrics from all environments between logging
        # These accumulate values and are cleared after each log
        self.episode_rewards: list[float] = []  # Currently unused, reserved for future
        self.episode_lengths: list[int] = []  # Currently unused, reserved for future
        self.badges_collected: list[int] = []  # Badge counts from info dicts
        self.events_triggered: list[int] = []  # Event flag counts from info dicts
        self.unique_coords: list[int] = []  # Exploration tile counts from info dicts

    def _on_step(self) -> bool:
        """
        Called after each training step. Collects metrics and logs periodically.

        Extracts game-specific metrics from the info dictionaries returned by
        each environment, buffers them, and logs aggregated statistics at
        regular intervals.

        Returns:
            True to continue training (never stops training).
        """
        # Collect info from all environments in the vectorized setup
        # self.locals contains variables from the training loop, including "infos"
        for info in self.locals.get("infos", []):
            # Extract metrics if present in the info dict
            # Not all steps have all metrics (only on episode boundaries)
            if "badges" in info:
                self.badges_collected.append(info["badges"])
            if "events" in info:
                self.events_triggered.append(info["events"])
            if "unique_coords" in info:
                self.unique_coords.append(info["unique_coords"])

        # Log aggregated metrics at regular intervals
        if self.num_timesteps % self.log_freq == 0 and self.badges_collected:
            # Log badge statistics
            # Mean shows typical performance, max shows best achieved
            self.logger.record("game/badges_mean", np.mean(self.badges_collected))
            self.logger.record("game/badges_max", max(self.badges_collected))

            # Log event and exploration statistics
            self.logger.record("game/events_mean", np.mean(self.events_triggered))
            self.logger.record("game/explore_tiles_mean", np.mean(self.unique_coords))

            # Clear buffers to track recent performance only
            # This prevents old data from affecting current statistics
            self.badges_collected.clear()
            self.events_triggered.clear()
            self.unique_coords.clear()

        return True


# =============================================================================
# STALL DETECTION CALLBACK
# =============================================================================


class StallDetectionCallback(BaseCallback):
    """
    Detect when training progress has stalled.

    Training can sometimes get stuck with the agent repeating ineffective
    behaviors. This callback monitors reward trends and warns when no
    improvement is detected for a specified number of steps.

    This is especially useful for Pokemon Red where:
    - The agent might get stuck in a corner
    - The agent might learn a local optimum (e.g., just walking in circles)
    - Training might destabilize after a hyperparameter becomes suboptimal

    Attributes:
        check_freq (int): Check for stalls every N steps.
        stall_threshold (int): Steps without progress before warning.
        min_progress (float): Minimum reward improvement to count as progress.
        last_reward_mean (float): Reward mean at last progress.
        steps_without_progress (int): Counter of steps since last progress.

    Example:
        >>> callback = StallDetectionCallback(
        ...     check_freq=10_000,
        ...     stall_threshold=50_000,
        ... )
        >>> # Will print warning if no reward improvement for 50K steps

    Notes:
        - Only warns, doesn't automatically intervene
        - Progress is measured by episode reward mean improvement
        - Useful for monitoring long training runs
    """

    def __init__(
        self,
        check_freq: int = 10_000,
        stall_threshold: int = 50_000,
        min_progress: float = 0.01,
        verbose: int = 1,
    ):
        """
        Initialize the stall detection callback.

        Args:
            check_freq: Check for stalls every N training steps.
                       10,000 is frequent enough to catch stalls early.
            stall_threshold: Number of steps without progress before warning.
                            50,000 steps allows for normal reward variance.
            min_progress: Minimum improvement in mean reward to count as progress.
                         0.01 filters out noise while detecting real improvement.
            verbose: Verbosity level. 1=print warnings when stalls detected.
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.stall_threshold = stall_threshold
        self.min_progress = min_progress

        # Tracking state
        self.last_check_step = 0  # Step count at last check
        self.last_reward_mean = float("-inf")  # Best reward mean seen
        self.steps_without_progress = 0  # Steps since last improvement

    def _on_step(self) -> bool:
        """
        Called after each training step. Checks for stalls periodically.

        Compares current mean episode reward to the best seen so far.
        If no improvement for stall_threshold steps, prints a warning.

        Returns:
            True to continue training (never stops training due to stalls).
        """
        # Only check at specified intervals (not every step for efficiency)
        if self.num_timesteps - self.last_check_step < self.check_freq:
            return True

        self.last_check_step = self.num_timesteps

        # Get recent episode rewards from the model's internal buffer
        # ep_info_buffer stores info from completed episodes
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            # Extract rewards from episode info dictionaries
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            current_mean = np.mean(rewards)

            # Check if we've made progress (reward improved by at least min_progress)
            if current_mean > self.last_reward_mean + self.min_progress:
                # Progress detected - reset counter and update best reward
                self.steps_without_progress = 0
                self.last_reward_mean = current_mean
            else:
                # No progress - increment counter
                self.steps_without_progress += self.check_freq

            # Warn if stalled for too long
            if self.steps_without_progress >= self.stall_threshold:
                if self.verbose:
                    print(
                        f"WARNING: No progress for {self.steps_without_progress:,} steps. "
                        f"Mean reward: {current_mean:.2f}"
                    )

        return True


# =============================================================================
# PERFORMANCE MONITORING CALLBACK
# =============================================================================


class PerformanceCallback(BaseCallback):
    """
    Monitor training speed in steps per second.

    Training speed varies based on:
    - Number of parallel environments
    - Hardware (CPU/GPU)
    - Game complexity (battles are slower than walking)
    - System load from other processes

    This callback tracks and logs steps/second to help identify:
    - Performance regressions
    - Bottlenecks in training
    - Optimal environment count for your hardware

    Attributes:
        log_freq (int): Log performance every N steps.
        last_time (float): Timestamp at last log.
        last_steps (int): Step count at last log.

    Example:
        >>> callback = PerformanceCallback(log_freq=10_000)
        >>> # Will print: "Step 10,000: 1234 steps/sec"

    Notes:
        - Performance is logged to both console (if verbose) and TensorBoard
        - Steps/second includes all parallel environments
        - Typical values: 500-2000 steps/sec depending on hardware
    """

    def __init__(self, log_freq: int = 10_000, verbose: int = 1):
        """
        Initialize the performance monitoring callback.

        Args:
            log_freq: Log performance metrics every N training steps.
                     10,000 steps provides regular updates without spam.
            verbose: Verbosity level. 1=print performance to console.
        """
        super().__init__(verbose)
        self.log_freq = log_freq

        # Timing state
        self.last_time = time.time()  # Timestamp at last measurement
        self.last_steps = 0  # Step count at last measurement

    def _on_step(self) -> bool:
        """
        Called after each training step. Logs performance at intervals.

        Calculates steps per second since the last measurement and logs
        to both TensorBoard and console.

        Returns:
            True to continue training (never stops training).
        """
        # Check if we've reached the logging interval
        if self.num_timesteps - self.last_steps >= self.log_freq:
            # Calculate elapsed time and steps
            current_time = time.time()
            elapsed = current_time - self.last_time
            steps_done = self.num_timesteps - self.last_steps

            # Calculate steps per second
            # max(elapsed, 1e-6) prevents division by zero if time is very short
            sps = steps_done / max(elapsed, 1e-6)

            # Log to TensorBoard
            self.logger.record("performance/steps_per_second", sps)

            # Print to console if verbose
            if self.verbose:
                print(f"Step {self.num_timesteps:,}: {sps:.0f} steps/sec")

            # Update tracking state for next measurement
            self.last_time = current_time
            self.last_steps = self.num_timesteps

        return True
