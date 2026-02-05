"""
Curriculum learning system for KantoRL.

This module provides auto-checkpointing curriculum learning that allows the
agent to train from progressively harder starting points. Instead of always
starting from the initial save state, the agent periodically saves game
states at milestones (new badges, significant event progress) and loads
them on reset, creating a natural curriculum.

Architecture Role:
    The curriculum system has two main components:
    1. CheckpointPool: Manages a directory of saved PyBoy game states
    2. CurriculumWrapper: Gymnasium wrapper that intercepts reset/step

    The flow is:
    make_env() → KantoRedEnv → CurriculumWrapper → StreamWrapper (optional)

    CurriculumWrapper sits between the base environment and any streaming
    wrapper, intercepting:
    - reset(): Loads a random checkpoint from the pool (75% of the time)
    - step(): Saves checkpoints at milestones, applies dynamic episode length

Key Design Decisions:
    - Atomic writes: State files use write-to-tmp + os.replace() for safety
    - Progress scoring: badges * 1000 + event_count (badges dominate)
    - Weighted sampling: Higher-progress checkpoints are sampled more often
    - Pool pruning: Lowest-progress checkpoints removed when pool exceeds max
    - Dynamic episode length: Scales with badge count (+20% per badge)

Dependencies:
    - gymnasium: For the Gymnasium wrapper base class
    - kantorl.rewards: For GameState and reward function syncing
    - kantorl.hm_automation: For automatic HM teaching and usage
    - kantorl.memory: For reading game state from emulator memory
    - kantorl.config: For KantoConfig curriculum settings
"""

import json
import os
import random
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import gymnasium as gym
import numpy as np

from kantorl.config import KantoConfig
from kantorl.hm_automation import HMAutomation
from kantorl.rewards import GameState

if TYPE_CHECKING:
    from kantorl.env import KantoRedEnv

# =============================================================================
# CHECKPOINT POOL
# =============================================================================


class CheckpointPool:
    """
    Manages a directory of saved PyBoy game states for curriculum learning.

    The pool stores game states as binary files with JSON metadata sidecars.
    States are named with their progress score for easy sorting and pruning.
    The pool automatically prunes lowest-progress states when it exceeds
    the maximum size.

    File Layout:
        pool_dir/
        ├── cp_001142_a3f8b2c1.state     # PyBoy save state binary
        ├── cp_001142_a3f8b2c1.json      # Metadata (badges, events, map_id)
        ├── cp_002305_7e4d1f9a.state
        ├── cp_002305_7e4d1f9a.json
        └── ...

    Naming Convention:
        cp_{progress_score:06d}_{uuid4_hex[:8]}.state
        - Progress score zero-padded to 6 digits for natural sorting
        - UUID suffix prevents collisions from multiple parallel environments

    Attributes:
        pool_dir: Path to the checkpoint directory.
        max_size: Maximum number of checkpoints to keep.

    Example:
        >>> pool = CheckpointPool(Path("runs/checkpoint_pool"), max_size=50)
        >>> pool.add(state_bytes, badges=2, event_count=142, map_id=3)
        >>> state_bytes, metadata = pool.sample()

    Notes:
        - Thread/process safe via atomic file writes (os.replace)
        - Shared across all SubprocVecEnv workers (filesystem-based)
        - Metadata includes badges, events, map_id, and timestamp
    """

    def __init__(self, pool_dir: Path, max_size: int = 50) -> None:
        """
        Initialize the checkpoint pool.

        Args:
            pool_dir: Directory to store checkpoint files. Created if needed.
            max_size: Maximum number of checkpoints. Oldest/lowest are pruned.
        """
        self.pool_dir = pool_dir
        self.max_size = max_size
        self.pool_dir.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        state_bytes: bytes,
        badges: int,
        event_count: int,
        map_id: int,
    ) -> None:
        """
        Save a game state to the pool with metadata.

        Uses atomic writes to prevent corruption: writes to a .tmp file
        first, then uses os.replace() which is atomic on both POSIX and
        Windows (NTFS).

        Args:
            state_bytes: Raw PyBoy save state bytes.
            badges: Number of badges at time of save (0-8).
            event_count: Number of event flags set.
            map_id: Current map ID when state was saved.

        Notes:
            - Progress score = badges * 1000 + event_count
            - UUID suffix prevents filename collisions across parallel envs
            - Prunes lowest-progress checkpoints if pool exceeds max_size
        """
        progress = badges * 1000 + event_count
        uid = uuid4().hex[:8]
        base_name = f"cp_{progress:06d}_{uid}"

        state_path = self.pool_dir / f"{base_name}.state"
        meta_path = self.pool_dir / f"{base_name}.json"

        # Atomic write: write to .tmp, then rename
        tmp_state = self.pool_dir / f"{base_name}.state.tmp"
        tmp_meta = self.pool_dir / f"{base_name}.json.tmp"

        # Write state binary
        with open(tmp_state, "wb") as f:
            f.write(state_bytes)
        os.replace(str(tmp_state), str(state_path))

        # Write metadata JSON
        metadata = {
            "badges": badges,
            "event_count": event_count,
            "map_id": map_id,
            "progress": progress,
        }
        with open(tmp_meta, "w") as f:
            json.dump(metadata, f)
        os.replace(str(tmp_meta), str(meta_path))

        # Prune if over max size
        self._prune()

    def sample(self) -> tuple[bytes, dict[str, Any]] | None:
        """
        Sample a checkpoint from the pool, weighted toward higher progress.

        Uses linear weighting: the highest-progress checkpoint is sampled
        most often, the lowest least often. This naturally creates a
        curriculum where the agent practices more from advanced states.

        Returns:
            Tuple of (state_bytes, metadata_dict) or None if pool is empty.
            Metadata contains: badges, event_count, map_id, progress.

        Notes:
            - Weights are proportional to rank (not progress score)
            - This prevents one very high checkpoint from dominating
            - Returns None if the pool directory is empty
        """
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None

        # Weight by rank (1-indexed): highest progress gets highest weight
        weights = list(range(1, len(checkpoints) + 1))
        total = sum(weights)
        probs = [w / total for w in weights]

        chosen = random.choices(checkpoints, weights=probs, k=1)[0]

        # Read state and metadata
        state_path = chosen
        meta_path = chosen.with_suffix(".json")

        with open(state_path, "rb") as f:
            state_bytes = f.read()

        metadata: dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        return state_bytes, metadata

    def best_progress(self) -> int:
        """
        Return the highest progress score in the pool.

        Returns:
            Highest progress score, or 0 if pool is empty.
        """
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return 0
        return self._get_progress(checkpoints[-1])

    def size(self) -> int:
        """Return the number of checkpoints in the pool."""
        return len(self._list_checkpoints())

    def _list_checkpoints(self) -> list[Path]:
        """
        List all checkpoint files sorted by progress score (ascending).

        Returns:
            Sorted list of .state file paths.
        """
        states = sorted(self.pool_dir.glob("cp_*.state"))
        return states

    def _get_progress(self, path: Path) -> int:
        """
        Extract progress score from checkpoint filename.

        Args:
            path: Checkpoint file path (e.g., cp_001142_a3f8b2c1.state).

        Returns:
            Progress score as integer, or 0 if parsing fails.
        """
        try:
            # Filename: cp_001142_a3f8b2c1.state → split("_")[1] = "001142"
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0

    def _prune(self) -> None:
        """
        Remove lowest-progress checkpoints when pool exceeds max_size.

        Removes the state file and its metadata sidecar together.
        """
        checkpoints = self._list_checkpoints()
        while len(checkpoints) > self.max_size:
            # Remove lowest-progress checkpoint (first in sorted list)
            to_remove = checkpoints.pop(0)
            to_remove.unlink(missing_ok=True)
            to_remove.with_suffix(".json").unlink(missing_ok=True)


# =============================================================================
# CURRICULUM WRAPPER
# =============================================================================


class CurriculumWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that adds curriculum learning to KantoRedEnv.

    This wrapper intercepts reset() and step() to provide:
    1. Checkpoint loading on reset (75% checkpoint, 25% fresh start)
    2. Milestone checkpoint saving during play
    3. Dynamic episode length scaling with badge progress
    4. HM automation (auto-teach and auto-use field moves)
    5. Reward function syncing to prevent spurious rewards

    Wrapping Order:
        KantoRedEnv → CurriculumWrapper → StreamWrapper (optional)
        The curriculum wrapper must come before StreamWrapper because it
        modifies the game state (loading checkpoints, HM memory writes).

    Attributes:
        pool: CheckpointPool instance for managing saved states.
        config: KantoConfig with curriculum settings.
        hm: HMAutomation instance for field move automation.
        checkpoint_weight: Probability of loading a checkpoint on reset.
        milestone_event_delta: Minimum event increase for milestone saves.
        _best_badges: Best badge count seen this episode (for milestone detection).
        _best_events: Best event count seen this episode.
        _step_count: Steps taken in current episode.
        _checkpoint_resets: Count of resets from checkpoints (for logging).
        _total_resets: Total reset count (for logging).

    Example:
        >>> env = KantoRedEnv(config=cfg, reward_fn="default")
        >>> env = CurriculumWrapper(env, pool_dir=Path("runs/pool"))
        >>> obs, info = env.reset()
        >>> # info may contain "curriculum_checkpoint": True

    Notes:
        - Must be applied after env.reset() has been called at least once
        - The pool directory is shared across all parallel environments
        - HM automation is tightly coupled to curriculum (enabled together)
    """

    def __init__(
        self,
        env: gym.Env,
        pool_dir: Path,
        config: KantoConfig | None = None,
    ) -> None:
        """
        Initialize the curriculum wrapper.

        Args:
            env: The base KantoRedEnv to wrap.
            pool_dir: Directory for the checkpoint pool.
            config: KantoConfig with curriculum settings. If None, uses defaults.
        """
        super().__init__(env)

        cfg = config or KantoConfig()

        self.pool = CheckpointPool(pool_dir, max_size=cfg.max_pool_size)
        self.config = cfg
        self.checkpoint_weight = cfg.checkpoint_weight
        self.milestone_event_delta = cfg.milestone_event_delta

        # HM automation instance (uses the unwrapped env's pyboy)
        self.hm = HMAutomation(self._base_env.pyboy)

        # Milestone tracking (reset each episode)
        self._best_badges = 0
        self._best_events = 0
        self._step_count = 0

        # Logging counters (persist across episodes)
        self._checkpoint_resets = 0
        self._total_resets = 0

    @property
    def _base_env(self) -> "KantoRedEnv":
        """Get the underlying KantoRedEnv (typed cast for attribute access)."""
        from kantorl.env import KantoRedEnv

        return cast(KantoRedEnv, self.unwrapped)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment, potentially loading a checkpoint.

        On each reset:
        1. Call base env.reset() normally (loads initial state, clears tracking)
        2. Roll dice: checkpoint_weight% checkpoint, rest fresh start
        3. If checkpoint selected and pool has states:
           - Load state into PyBoy
           - Sync reward function to prevent spurious rewards
           - Rebuild observation and info

        Args:
            seed: Random seed (passed to base env).
            options: Additional options (passed to base env).

        Returns:
            Tuple of (observation, info) with curriculum metadata.
        """
        # Always call base reset first (initializes emulator, clears tracking)
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset episode tracking
        self._step_count = 0
        self._best_badges = 0
        self._best_events = 0
        self._total_resets += 1

        # Reset HM automation
        self.hm.reset()

        # Decide whether to load a checkpoint
        use_checkpoint = random.random() < self.checkpoint_weight
        loaded = False

        if use_checkpoint:
            result = self.pool.sample()
            if result is not None:
                state_bytes, metadata = result

                # Load the checkpoint state into PyBoy
                self._base_env.pyboy.load_state(BytesIO(state_bytes))

                # Sync reward function to prevent spurious rewards
                # Create a GameState from the loaded checkpoint's metadata
                checkpoint_state = GameState(
                    badges=metadata.get("badges", 0),
                    event_count=metadata.get("event_count", 0),
                    map_id=metadata.get("map_id", 0),
                )
                self._base_env.reward_fn.sync_to_state(checkpoint_state)

                # Update milestone tracking to match checkpoint
                self._best_badges = metadata.get("badges", 0)
                self._best_events = metadata.get("event_count", 0)

                # Rebuild observation from the loaded state
                obs = self._base_env._get_observation()
                info = self._base_env._get_info()

                loaded = True
                self._checkpoint_resets += 1

        # Add curriculum metadata to info
        info["curriculum_checkpoint"] = loaded
        info["curriculum_pool_size"] = self.pool.size()
        info["curriculum_best_progress"] = self.pool.best_progress()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Execute one step with HM automation, milestone saving, and dynamic length.

        On each step:
        1. Run HM automation (auto-teach/use field moves)
        2. Call base env.step() normally
        3. Check for milestones (new badge or event delta exceeded)
        4. Apply dynamic episode length (if enabled)
        5. Add curriculum metadata to info

        Args:
            action: Action index (0-7).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            with curriculum metadata added to info.
        """
        # Run HM automation before the step
        hm_info = self.hm.step(self._step_count)

        # Execute the base environment step
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        # -----------------------------------------------------------------
        # Milestone Detection and Checkpoint Saving
        # -----------------------------------------------------------------
        current_badges = info.get("badges", 0)
        current_events = info.get("events", 0)

        is_milestone = False

        # New badge earned
        if current_badges > self._best_badges:
            is_milestone = True
            self._best_badges = current_badges

        # Significant event progress
        if current_events >= self._best_events + self.milestone_event_delta:
            is_milestone = True
            self._best_events = current_events

        # Save checkpoint at milestones
        if is_milestone:
            state_buffer = BytesIO()
            self._base_env.pyboy.save_state(state_buffer)
            map_id = info.get("map_id", 0)
            self.pool.add(
                state_bytes=state_buffer.getvalue(),
                badges=current_badges,
                event_count=current_events,
                map_id=map_id,
            )

        # -----------------------------------------------------------------
        # Dynamic Episode Length
        # -----------------------------------------------------------------
        if self.config.dynamic_episode_length:
            # Scale max_steps by +20% per badge
            # At 0 badges: max_steps * 1.0
            # At 8 badges: max_steps * 2.6
            scale = 1.0 + 0.2 * current_badges
            dynamic_max = int(self.config.max_steps * scale)
            truncated = self._step_count >= dynamic_max

        # -----------------------------------------------------------------
        # Add Curriculum Metadata to Info
        # -----------------------------------------------------------------
        info["curriculum_checkpoint"] = False  # Not a reset step
        info["curriculum_pool_size"] = self.pool.size()
        info["curriculum_best_progress"] = self.pool.best_progress()
        info["curriculum_dynamic_max_steps"] = (
            int(self.config.max_steps * (1.0 + 0.2 * current_badges))
            if self.config.dynamic_episode_length
            else self.config.max_steps
        )
        info["curriculum_hm_taught"] = hm_info.get("hm_taught", [])
        info["curriculum_hm_used"] = hm_info.get("hm_used", [])
        info["curriculum_hm_all_taught"] = self.hm.get_taught_moves()
        info["curriculum_checkpoint_resets"] = self._checkpoint_resets
        info["curriculum_total_resets"] = self._total_resets

        return obs, reward, terminated, truncated, info
