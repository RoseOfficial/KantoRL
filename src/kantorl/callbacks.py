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

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecNormalize

    import wandb as wandb_mod


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
        vec_normalize: VecNormalize | None = None,
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
            vec_normalize: Optional VecNormalize wrapper whose running statistics
                          (obs mean/var, reward mean/var) are saved alongside each
                          model checkpoint. Required for correct reward scaling on
                          resume.
        """
        super().__init__(verbose)

        # Convert string path to Path object for consistent handling
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self.vec_normalize = vec_normalize

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

            # Save VecNormalize running statistics alongside the model
            # These are needed to restore reward/obs normalization on resume
            if self.vec_normalize is not None:
                vecnorm_path = self.save_path / "vecnormalize.pkl"
                self.vec_normalize.save(str(vecnorm_path))

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

    def __init__(
        self,
        log_freq: int = 1000,
        session_path: Path | None = None,
        verbose: int = 0,
    ):
        """
        Initialize the TensorBoard callback.

        Args:
            log_freq: Log aggregated metrics every N training steps.
                     1000 steps provides good granularity without too much data.
            session_path: Directory for session outputs (event checklist file).
                         If None, checklist file is not written.
            verbose: Verbosity level (currently unused, for API consistency).
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.session_path = session_path

        # Buffers to collect metrics from all environments between logging
        # These accumulate values and are cleared after each log
        self.episode_rewards: list[float] = []  # Currently unused, reserved for future
        self.episode_lengths: list[int] = []  # Currently unused, reserved for future
        self.badges_collected: list[int] = []  # Badge counts from info dicts
        self.events_triggered: list[int] = []  # Event flag counts from info dicts
        self.unique_coords: list[int] = []  # Exploration tile counts from info dicts

        # Full event tracking — persists entire training session
        # Union of all event flag bit indices seen across all envs, all time
        self._all_events_seen: set[int] = set()
        # Maps bit indices to human-readable names (loaded lazily on first step)
        self._milestone_map: dict[int, str] = {}

        # Curriculum learning metric buffers
        # Only populated when curriculum mode is active
        self.curriculum_pool_sizes: list[int] = []
        self.curriculum_best_progress: list[int] = []
        self.curriculum_checkpoint_resets: list[int] = []
        self.curriculum_total_resets: list[int] = []
        self.curriculum_dynamic_max_steps: list[int] = []

    def _on_step(self) -> bool:
        """
        Called after each training step. Collects metrics and logs periodically.

        Extracts game-specific metrics from the info dictionaries returned by
        each environment, buffers them, and logs aggregated statistics at
        regular intervals.

        Returns:
            True to continue training (never stops training).
        """
        # Load milestone name map once (lazy — avoids import at module level)
        if not self._milestone_map:
            from kantorl.env import _load_milestones

            self._milestone_map = _load_milestones()

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

            # Collect newly set event flags and announce first-time events
            for idx in info.get("new_event_indices", []):
                if idx not in self._all_events_seen:
                    self._all_events_seen.add(idx)
                    name = self._milestone_map.get(idx, f"Event #{idx}")
                    print(f"  EVENT: {name}")

            # Collect curriculum metrics (only present when curriculum is active)
            if "curriculum_pool_size" in info:
                self.curriculum_pool_sizes.append(info["curriculum_pool_size"])
            if "curriculum_best_progress" in info:
                self.curriculum_best_progress.append(info["curriculum_best_progress"])
            if "curriculum_checkpoint_resets" in info:
                self.curriculum_checkpoint_resets.append(info["curriculum_checkpoint_resets"])
            if "curriculum_total_resets" in info:
                self.curriculum_total_resets.append(info["curriculum_total_resets"])
            if "curriculum_dynamic_max_steps" in info:
                self.curriculum_dynamic_max_steps.append(info["curriculum_dynamic_max_steps"])

        # Log aggregated metrics at regular intervals
        if self.num_timesteps % self.log_freq == 0 and self.badges_collected:
            # Log badge statistics
            # Mean shows typical performance, max shows best achieved
            self.logger.record("game/badges_mean", np.mean(self.badges_collected))
            self.logger.record("game/badges_max", max(self.badges_collected))

            # Log event and exploration statistics
            self.logger.record("game/events_mean", np.mean(self.events_triggered))
            self.logger.record("game/explore_tiles_mean", np.mean(self.unique_coords))

            # Log total unique events ever seen across all envs (session-wide)
            self.logger.record("game/events_achieved", len(self._all_events_seen))

            # Log curriculum metrics (only if curriculum data was collected)
            if self.curriculum_pool_sizes:
                self.logger.record(
                    "curriculum/pool_size",
                    max(self.curriculum_pool_sizes),
                )
            if self.curriculum_best_progress:
                self.logger.record(
                    "curriculum/best_progress",
                    max(self.curriculum_best_progress),
                )
            if self.curriculum_checkpoint_resets and self.curriculum_total_resets:
                # Calculate percentage of resets that used checkpoints
                total_cp = max(self.curriculum_checkpoint_resets)
                total_all = max(max(self.curriculum_total_resets), 1)
                self.logger.record(
                    "curriculum/checkpoint_reset_pct",
                    total_cp / total_all,
                )
            if self.curriculum_dynamic_max_steps:
                self.logger.record(
                    "curriculum/dynamic_max_steps",
                    np.mean(self.curriculum_dynamic_max_steps),
                )

            # Write event checklist file (JSON snapshot of all events seen)
            if self.session_path is not None:
                self._write_event_checklist()

            # Clear buffers to track recent performance only
            # This prevents old data from affecting current statistics
            self.badges_collected.clear()
            self.events_triggered.clear()
            self.unique_coords.clear()
            self.curriculum_pool_sizes.clear()
            self.curriculum_best_progress.clear()
            self.curriculum_checkpoint_resets.clear()
            self.curriculum_total_resets.clear()
            self.curriculum_dynamic_max_steps.clear()

        return True

    def _write_event_checklist(self) -> None:
        """
        Write full event flag checklist to a JSON file in the session directory.

        Produces a snapshot of all 2560 event flags: which named events exist,
        which have been achieved, and which unnamed bit indices were triggered.
        This file is overwritten each log interval so it always reflects the
        latest state.

        The output file is ``session_path/event_checklist.json``.
        """
        import json

        assert self.session_path is not None  # guarded by caller
        checklist_path = self.session_path / "event_checklist.json"

        # Named events section — one entry per curated milestone from events.json
        named_events: dict[str, dict[str, object]] = {}
        for idx in sorted(self._milestone_map):
            name = self._milestone_map[idx]
            named_events[str(idx)] = {
                "name": name,
                "achieved": idx in self._all_events_seen,
            }

        # Unnamed achieved — bit indices that fired but aren't in events.json
        unnamed_achieved = sorted(
            idx for idx in self._all_events_seen if idx not in self._milestone_map
        )

        checklist: dict[str, object] = {
            "step": self.num_timesteps,
            "total_achieved": len(self._all_events_seen),
            "total_flags": 2560,
            "named_achieved": sum(
                1 for idx in self._milestone_map if idx in self._all_events_seen
            ),
            "named_total": len(self._milestone_map),
            "named_events": named_events,
            "unnamed_achieved": unnamed_achieved,
        }

        with open(checklist_path, "w") as f:
            json.dump(checklist, f, indent=2)


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


# =============================================================================
# WANDB LOGGING CALLBACK
# =============================================================================


class WandbCallback(BaseCallback):
    """
    Log game metrics and PPO losses to Weights & Biases.

    Runs alongside TensorboardCallback to provide the same game-specific
    metrics (badges, events, exploration) plus PPO loss metrics to wandb
    for experiment tracking and comparison dashboards.

    Metric Hierarchy:
        - game/*: Pokemon Red progress (badges_mean, events_mean, explore_tiles_mean)
        - losses/*: PPO training losses (policy_loss, value_loss, entropy_loss, etc.)
        - performance/*: Training speed (sps = steps per second)
        - curriculum/*: Curriculum learning stats (pool_size, best_progress, etc.)

    Design Decisions:
        - Time-throttled logging (every N seconds) instead of step-based,
          to avoid overwhelming the wandb API with high-frequency uploads.
          This matches the pattern used by pokemonred_puffer.
        - Collects metrics from self.locals["infos"] (same source as
          TensorboardCallback) and self.logger.name_to_value (PPO losses).
        - Does NOT replace TensorBoard — both run simultaneously.

    Attributes:
        wandb_run: Reference to the active wandb run for logging.
        log_interval_sec: Minimum seconds between wandb.log() calls.
        last_log_time: Timestamp of the last wandb.log() call.

    Example:
        >>> import wandb
        >>> run = wandb.init(project="kantorl")
        >>> callback = WandbCallback(wandb_run=run, log_interval_sec=5.0)
    """

    def __init__(
        self,
        wandb_run: wandb_mod.sdk.wandb_run.Run,
        log_interval_sec: float = 5.0,
        verbose: int = 0,
    ) -> None:
        """
        Initialize the wandb callback.

        Args:
            wandb_run: Active wandb run object from wandb.init().
                       Used for logging metrics via wandb_run.log().
            log_interval_sec: Minimum seconds between wandb.log() calls.
                             5.0 seconds is a good balance: frequent enough
                             for live dashboard updates, infrequent enough to
                             avoid API rate limits.
            verbose: Verbosity level (0=silent, 1=print log events).
        """
        super().__init__(verbose)
        self.wandb_run = wandb_run
        self.log_interval_sec = log_interval_sec
        self.last_log_time = time.time()

        # Metric buffers — accumulate between log calls, then aggregate
        self.badges_buffer: list[int] = []
        self.events_buffer: list[int] = []
        self.coords_buffer: list[int] = []
        self.curriculum_pool_sizes: list[int] = []
        self.curriculum_best_progress: list[int] = []

        # Full event tracking — persists entire training session
        self._all_events_seen: set[int] = set()
        self._milestone_map: dict[int, str] = {}

    def _on_step(self) -> bool:
        """
        Called after each training step. Collects metrics and logs to wandb.

        This method:
        1. Collects game metrics from info dicts (every step)
        2. Checks if enough time has elapsed since last log
        3. If so, aggregates buffered metrics + grabs PPO losses + logs to wandb

        Returns:
            True to continue training (never stops training).
        """
        # Load milestone name map once (lazy — avoids import at module level)
        if not self._milestone_map:
            from kantorl.env import _load_milestones

            self._milestone_map = _load_milestones()

        # -----------------------------------------------------------------
        # Phase 1: Collect metrics from every env (runs every step)
        # -----------------------------------------------------------------
        # self.locals["infos"] is a list with one info dict per parallel env.
        # Each info dict contains game-state metrics set by KantoRedEnv.step().
        for info in self.locals.get("infos", []):
            if "badges" in info:
                self.badges_buffer.append(info["badges"])
            if "events" in info:
                self.events_buffer.append(info["events"])
            if "unique_coords" in info:
                self.coords_buffer.append(info["unique_coords"])
            # Collect newly set event flags (no console announcement — wandb only)
            for idx in info.get("new_event_indices", []):
                self._all_events_seen.add(idx)
            # Curriculum metrics (only present when curriculum mode is active)
            if "curriculum_pool_size" in info:
                self.curriculum_pool_sizes.append(info["curriculum_pool_size"])
            if "curriculum_best_progress" in info:
                self.curriculum_best_progress.append(info["curriculum_best_progress"])

        # -----------------------------------------------------------------
        # Phase 2: Ship to wandb if enough time has elapsed
        # -----------------------------------------------------------------
        now = time.time()
        elapsed = now - self.last_log_time
        if elapsed < self.log_interval_sec or not self.badges_buffer:
            return True

        # Build the metrics dict — game progress
        metrics: dict[str, float | int] = {
            "game/badges_mean": float(np.mean(self.badges_buffer)),
            "game/events_mean": float(np.mean(self.events_buffer))
            if self.events_buffer
            else 0.0,
            "game/explore_tiles_mean": float(np.mean(self.coords_buffer))
            if self.coords_buffer
            else 0.0,
            "game/events_achieved": len(self._all_events_seen),
        }

        # PPO losses — only populated after an optimizer update, not every step.
        # self.logger.name_to_value is SB3's internal metric store.
        loss_map = {
            "train/policy_gradient_loss": "losses/policy_loss",
            "train/value_loss": "losses/value_loss",
            "train/entropy_loss": "losses/entropy_loss",
            "train/approx_kl": "losses/approx_kl",
            "train/clip_fraction": "losses/clip_fraction",
        }
        for sb3_key, wandb_key in loss_map.items():
            value = self.logger.name_to_value.get(sb3_key)
            if value is not None:
                metrics[wandb_key] = float(value)

        # Training speed (steps per second since last log)
        steps_since = self.num_timesteps - getattr(self, "_last_log_steps", 0)
        metrics["performance/sps"] = steps_since / max(elapsed, 1e-6)
        self._last_log_steps = self.num_timesteps

        # Curriculum metrics
        if self.curriculum_pool_sizes:
            metrics["curriculum/pool_size"] = max(self.curriculum_pool_sizes)
        if self.curriculum_best_progress:
            metrics["curriculum/best_progress"] = max(self.curriculum_best_progress)

        # Single wandb.log() call with all metrics for this interval
        self.wandb_run.log(metrics, step=self.num_timesteps)

        # Reset buffers and timer for next interval
        # Note: _all_events_seen is NOT cleared — it persists for the session
        self.last_log_time = now
        self.badges_buffer.clear()
        self.events_buffer.clear()
        self.coords_buffer.clear()
        self.curriculum_pool_sizes.clear()
        self.curriculum_best_progress.clear()

        return True

    def _on_training_end(self) -> None:
        """
        Called when training finishes. Flushes any remaining buffered metrics.

        Ensures the final batch of collected metrics is logged to wandb
        even if the time interval hasn't elapsed yet. This prevents losing
        data from the last few seconds of training.
        """
        # Force-log any remaining buffered data
        if self.badges_buffer:
            metrics: dict[str, float | int] = {
                "game/badges_mean": float(np.mean(self.badges_buffer)),
                "game/events_mean": float(np.mean(self.events_buffer))
                if self.events_buffer
                else 0.0,
                "game/explore_tiles_mean": float(np.mean(self.coords_buffer))
                if self.coords_buffer
                else 0.0,
                "game/events_achieved": len(self._all_events_seen),
            }
            if self.curriculum_pool_sizes:
                metrics["curriculum/pool_size"] = max(self.curriculum_pool_sizes)
            if self.curriculum_best_progress:
                metrics["curriculum/best_progress"] = max(self.curriculum_best_progress)
            self.wandb_run.log(metrics, step=self.num_timesteps)
