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
import time
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
        ├── cp_001142_000000001234567890_a3f8b2c1.state  # PyBoy save state
        ├── cp_001142_000000001234567890_a3f8b2c1.json   # Metadata
        ├── cp_002305_000000001987654321_7e4d1f9a.state
        ├── cp_002305_000000001987654321_7e4d1f9a.json
        └── ...

    Naming Convention:
        cp_{progress_score:06d}_{save_serial:020d}_{uuid4_hex[:8]}.state
        - Progress score zero-padded for natural sorting
        - Save serial in the name so pruning does not read every sidecar
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
        step: int = 0,
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
            step: Monotonic save serial, used by recent-state sampling.

        Notes:
            - Progress score = badges * 1000 + event_count
            - UUID suffix prevents filename collisions across parallel envs
            - Pruning keeps both the highest-progress half and the most recent
              half, so a deep dungeon state is not discarded for an old badge
        """
        progress = badges * 1000 + event_count
        uid = uuid4().hex[:8]
        # The save serial is in the filename so pruning can sort by recency
        # without opening every metadata sidecar. 20 digits fits time_ns().
        base_name = f"cp_{progress:06d}_{int(step):020d}_{uid}"

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
            "step": int(step),
        }
        with open(tmp_meta, "w") as f:
            json.dump(metadata, f)
        os.replace(str(tmp_meta), str(meta_path))

        # Prune if over max size
        self._prune()

    def sample(
        self,
        frontier_fraction: float = 0.5,
        recent_fraction: float = 0.3,
    ) -> tuple[bytes, dict[str, Any]] | None:
        """
        Sample a checkpoint from the pool.

        The default mix is 50% frontier (highest-progress half), 30% recent
        (latest third by save serial) and 20% uniform. Frontier sampling gives
        the agent hard states to practise; recent sampling keeps the current
        map in rotation even when it has not produced a new event yet; uniform
        sampling prevents the pool from collapsing onto one state.

        Args:
            frontier_fraction: Probability of sampling the highest-progress half.
            recent_fraction: Probability of sampling the latest third by step.

        Returns:
            Tuple of (state_bytes, metadata_dict) or None if pool is empty.

        Notes:
            - Fractions must sum to at most 1.0 (validated in KantoConfig)
            - Returns None if the pool directory is empty
        """
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None

        roll = random.random()
        if roll < frontier_fraction:
            half = max(1, len(checkpoints) // 2)
            candidates = checkpoints[-half:]
        elif roll < frontier_fraction + recent_fraction:
            candidates = self._list_recent(max(1, len(checkpoints) // 3))
        else:
            candidates = checkpoints

        # A state can be pruned by another worker between listing and reading.
        # Retry with a fresh listing instead of failing the reset.
        for _ in range(3):
            if not candidates:
                return None
            chosen = random.choice(candidates)
            try:
                with open(chosen, "rb") as f:
                    state_bytes = f.read()
                metadata = json.loads(chosen.with_suffix(".json").read_text())
            except (OSError, json.JSONDecodeError):
                checkpoints = self._list_checkpoints()
                if not checkpoints:
                    return None
                candidates = checkpoints
                continue
            return state_bytes, metadata

        return None

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

    def _list_recent(self, count: int) -> list[Path]:
        """Return the ``count`` most recently saved checkpoints.

        Recency is the monotonic save serial in the metadata sidecar, not the
        progress score: a deep dungeon state with no new event flag is still
        the frontier and must stay in rotation.
        """
        checkpoints = self._list_checkpoints()
        ordered = sorted(checkpoints, key=self._get_step)
        return ordered[-count:] if count > 0 else []

    def _get_step(self, path: Path) -> int:
        """Read the save serial from the filename, falling back to metadata.

        New files carry ``cp_<progress>_<step>_<uid>``; older files have no
        step component and need the sidecar.
        """
        parts = path.stem.split("_")
        if len(parts) >= 4:
            try:
                return int(parts[2])
            except ValueError:
                pass
        meta_path = path.with_suffix(".json")
        try:
            with open(meta_path) as f:
                return int(json.load(f).get("step", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            return 0

    def _prune(self) -> None:
        """
        Keep the highest-progress half and the most recent half.

        Pruning on progress alone discards the deep dungeon state that has not
        triggered an event yet, which is exactly the state the archive exists
        to preserve. The union is at most ``max_size`` entries.

        Workers share this directory and may be reading a state when another
        worker prunes it, so deletion is best-effort. A small hysteresis avoids
        pruning on every add once the pool is full.
        """
        checkpoints = self._list_checkpoints()
        if len(checkpoints) <= self.max_size + 8:
            return

        half = max(1, self.max_size // 2)
        keep = set(checkpoints[-half:]) | set(self._list_recent(half))

        for path in checkpoints:
            if path not in keep:
                self._remove(path)

    def _remove(self, path: Path) -> None:
        """Best-effort deletion of a state and its metadata sidecar."""
        for target in (path, path.with_suffix(".json")):
            try:
                target.unlink(missing_ok=True)
            except OSError:
                # Another worker has the file open or already removed it.
                continue


# =============================================================================
# CURRICULUM WRAPPER
# =============================================================================


class CurriculumWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that adds curriculum learning to KantoRedEnv.

    This wrapper intercepts reset() and step() to provide:
    1. Checkpoint loading on reset (75% checkpoint, 25% fresh start)
    2. Milestone and frontier checkpoint saving during play
    3. Dynamic episode length scaling with badge progress
    4. Optional HM automation (off by default; it is hand-authored direction)
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
        - HM automation is opt-in via ``enable_hm_automation``
    """

    def __init__(
        self,
        env: gym.Env,
        pool_dir: Path,
        config: KantoConfig | None = None,
        initial_step_offset: int = 0,
    ) -> None:
        """
        Initialize the curriculum wrapper.

        Args:
            env: The base KantoRedEnv to wrap.
            pool_dir: Directory for the checkpoint pool.
            config: KantoConfig with curriculum settings. If None, uses defaults.
            initial_step_offset: Step counter offset for the first episode, used
                to stagger episode truncation across parallel environments so they
                don't all reset at the same global step.
        """
        super().__init__(env)

        cfg = config or KantoConfig()

        self.pool = CheckpointPool(pool_dir, max_size=cfg.max_pool_size)
        self.config = cfg
        self.checkpoint_weight = cfg.checkpoint_weight
        self.milestone_event_delta = cfg.milestone_event_delta

        # HM automation is hand-authored direction, so it is opt-in. The
        # archive itself (state pool + resets) is domain-agnostic.
        self.hm = (
            HMAutomation(self._base_env.pyboy)
            if cfg.enable_hm_automation
            else None
        )

        # Milestone tracking (reset each episode)
        self._best_badges = 0
        self._best_events = 0
        self._step_count = 0

        # Frontier tracking: save a state on a map change and every N new
        # tiles. The set is per episode; resets from the pool start a new one.
        self._visited_cells: set[tuple[int, int]] = set()
        self._last_map_id: int | None = None
        self._save_cooldown = 0
        self._save_serial = 0

        # Stagger offset: pre-advances step counter on first episode only
        # so parallel envs truncate at different times
        self._initial_step_offset = initial_step_offset
        self._first_episode = True

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

        # Apply stagger offset on first episode to prevent synchronized truncation
        # across all parallel envs. After the first episode, envs are naturally
        # desynchronized so no offset is needed.
        if self._first_episode and self._initial_step_offset > 0:
            self._step_count = self._initial_step_offset
            self._first_episode = False
        else:
            self._step_count = 0

        self._best_badges = 0
        self._best_events = 0
        self._total_resets += 1
        self._visited_cells.clear()
        self._last_map_id = None
        self._save_cooldown = 0

        # Reset HM automation when it is enabled
        if self.hm is not None:
            self.hm.reset()

        # Decide whether to load a checkpoint
        use_checkpoint = random.random() < self.checkpoint_weight
        loaded = False

        if use_checkpoint:
            result = self.pool.sample(
                frontier_fraction=self.config.archive_frontier_fraction,
                recent_fraction=self.config.archive_recent_fraction,
            )
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
        # Run HM automation before the step when it is enabled
        hm_info: dict[str, Any] = {}
        if self.hm is not None:
            hm_info = self.hm.step(self._step_count)

        # Execute the base environment step
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        if self._save_cooldown > 0:
            self._save_cooldown -= 1

        # -----------------------------------------------------------------
        # Milestone and Frontier Detection
        # -----------------------------------------------------------------
        current_badges = int(info.get("badges", 0))
        current_events = int(info.get("events", 0))
        map_id = int(info.get("map_id", 0))
        position = info.get("position")

        is_milestone = False

        # New badge earned
        if current_badges > self._best_badges:
            is_milestone = True
            self._best_badges = current_badges

        # Significant event progress
        if current_events >= self._best_events + self.milestone_event_delta:
            is_milestone = True
            self._best_events = current_events

        # Frontier save: a new map, or every N new tiles on the current map.
        # The cell set is map-local; two maps can share (x, y) coordinates.
        archive_trigger = False
        if position is not None:
            x, y = position
            cell = (int(x), int(y))
            if self._last_map_id is None:
                self._last_map_id = map_id
            elif map_id != self._last_map_id:
                self._last_map_id = map_id
                self._visited_cells.clear()
                archive_trigger = self.config.archive_on_new_map
            self._visited_cells.add(cell)
            stride = self.config.archive_cell_stride
            if stride > 0 and len(self._visited_cells) % stride == 0:
                archive_trigger = True

        if archive_trigger and self._save_cooldown > 0:
            archive_trigger = False

        archive_saved = is_milestone or archive_trigger
        if archive_saved:
            self._save_checkpoint(current_badges, current_events, map_id)

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
        info["curriculum_hm_all_taught"] = (
            self.hm.get_taught_moves() if self.hm is not None else []
        )
        info["curriculum_archive_saved"] = archive_saved
        info["curriculum_checkpoint_resets"] = self._checkpoint_resets
        info["curriculum_total_resets"] = self._total_resets

        return obs, reward, terminated, truncated, info

    def _save_checkpoint(self, badges: int, events: int, map_id: int) -> None:
        """Snapshot the current emulator state into the shared pool.

        Archive writes are best-effort: a failed save must not kill the
        training worker. The next frontier trigger will try again.
        """
        try:
            state_buffer = BytesIO()
            self._base_env.pyboy.save_state(state_buffer)
            self.pool.add(
                state_bytes=state_buffer.getvalue(),
                badges=badges,
                event_count=events,
                map_id=map_id,
                step=time.time_ns(),
            )
            self._save_cooldown = self.config.archive_save_cooldown
        except OSError:
            return
