"""Concrete observation producers.

Each class reads exactly one entry of the observation dict. The default set
reproduces the original hard-coded observation exactly; ``local_map`` is
opt-in:

====================  ==========================================  ==================
Key                   Meaning                                     Space
====================  ==========================================  ==================
``screens``           Stacked downscaled grayscale frames         ``(3, 72, 80)`` uint8
``health``            Party HP fraction                           ``(1,)`` float32
``level``             Fourier-encoded party levels                ``(8,)`` float32
``badges``            Per-badge binary flags                      ``(8,)`` int8
``events``            Full event-flag bitfield                    ``(2560,)`` int8
``map``               48x48 visited-coordinate bitmap             ``(48, 48)`` uint8
``local_map``         15x15 visited-tile window around player     ``(15, 15)`` uint8
``recent_actions``    Last three action indices                   ``(3,)`` int64
====================  ==========================================  ==================

Selecting a subset in ``KantoConfig.obs_components`` is how the tiers are
expressed: pixels-only, pixels + progress, or the full RAM-rich observation.
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from gymnasium import spaces

from kantorl.obs.base import ObsContext

# Number of past actions exposed to the policy.
ACTION_HISTORY_LEN = 3

# Size of the event-flag observation. ``memory.get_event_flags`` pads the
# 312-byte (2496-bit) bitfield to this size for alignment.
EVENT_OBS_DIM = 2560

# Downscaled exploration bitmap shape.
MAP_OBS_SHAPE = (48, 48)

# Local visited-tile window around the player. Unlike ``map`` (48x48 global
# cells, each ~9x9 tiles), this is one cell per tile and moves with the player,
# so indoor rooms and corridors keep their true shape.
LOCAL_MAP_SIZE = 15
LOCAL_MAP_CENTER = LOCAL_MAP_SIZE // 2
LOCAL_MAP_PLAYER = 128

# Number of party levels encoded by LevelProducer.
LEVEL_OBS_DIM = 8

_T = TypeVar("_T")


def _require(value: _T | None, name: str) -> _T:
    """Return a context field or fail loudly if the tier did not provide it."""
    if value is None:
        raise ValueError(
            f"Observation producer requires ctx.{name}, but it was not populated. "
            "This indicates a mismatch between the configured producers and the "
            "environment's context builder."
        )
    return value


class ScreenProducer:
    """Downscaled, frame-stacked grayscale screen."""

    key = "screens"

    def __init__(self, frame_stacks: int, screen_size: tuple[int, int]) -> None:
        self.frame_stacks = frame_stacks
        self.screen_size = screen_size

    def space(self) -> spaces.Space[Any]:
        height, width = self.screen_size
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.frame_stacks, height, width),
            dtype=np.uint8,
        )

    def reset(self) -> None:
        """No cached state; the environment owns the frame buffer."""

    def read(self, ctx: ObsContext) -> np.ndarray:
        return _require(ctx.frame_stack, "frame_stack")


class HealthProducer:
    """Party HP fraction in ``[0, 1]``."""

    key = "health"

    def space(self) -> spaces.Space[Any]:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self) -> None:
        """Stateless."""

    def read(self, ctx: ObsContext) -> np.ndarray:
        # max(..., 1) avoids division by zero before the first Pokemon.
        return np.array([ctx.current_hp / max(ctx.max_hp, 1)], dtype=np.float32)


class LevelProducer:
    """Fourier-encoded average party level.

    Levels change only on level-up, so the encoding is cached by the exact
    party-level tuple. This matches the original env's cache and keeps the
    per-step cost to a tuple comparison in the common case.
    """

    key = "level"

    def __init__(self, dim: int = LEVEL_OBS_DIM) -> None:
        if dim % 2 != 0:
            raise ValueError(f"Level encoding dim must be even, got {dim}")
        self.dim = dim
        self._cache_key: tuple[int, ...] = ()
        self._cache_val: np.ndarray = np.zeros(dim, dtype=np.float32)

    def space(self) -> spaces.Space[Any]:
        return spaces.Box(low=-1, high=1, shape=(self.dim,), dtype=np.float32)

    def reset(self) -> None:
        self._cache_key = ()
        self._cache_val = np.zeros(self.dim, dtype=np.float32)

    def read(self, ctx: ObsContext) -> np.ndarray:
        if not ctx.levels:
            return np.zeros(self.dim, dtype=np.float32)

        key = tuple(ctx.levels)
        if key == self._cache_key:
            return self._cache_val

        # Normalize average level to [0, 1], then encode with sin/cos pairs.
        avg_level = np.mean(ctx.levels) / 100.0
        freqs = np.arange(1, self.dim // 2 + 1)
        sin_features = np.sin(2 * np.pi * freqs * avg_level)
        cos_features = np.cos(2 * np.pi * freqs * avg_level)
        features = np.concatenate([sin_features, cos_features]).astype(np.float32)

        self._cache_key = key
        self._cache_val = features
        return features


class BadgesProducer:
    """Per-badge binary flags (Boulder through Earth)."""

    key = "badges"

    def space(self) -> spaces.Space[Any]:
        return spaces.MultiBinary(8)

    def reset(self) -> None:
        """Stateless."""

    def read(self, ctx: ObsContext) -> np.ndarray:
        return _require(ctx.badge_flags, "badge_flags")


class EventsProducer:
    """Full event-flag bitfield.

    This is the strongest prior in the default observation: it lets the policy
    read story progress directly instead of inferring it from pixels. Tiers
    that want to test genuine perception should omit ``"events"``.
    """

    key = "events"

    def space(self) -> spaces.Space[Any]:
        return spaces.MultiBinary(EVENT_OBS_DIM)

    def reset(self) -> None:
        """Stateless."""

    def read(self, ctx: ObsContext) -> np.ndarray:
        return _require(ctx.event_flags, "event_flags")


class MapProducer:
    """Downscaled bitmap of global coordinates visited this episode."""

    key = "map"

    def space(self) -> spaces.Space[Any]:
        return spaces.Box(low=0, high=255, shape=MAP_OBS_SHAPE, dtype=np.uint8)

    def reset(self) -> None:
        """No cached state; the environment owns the bitmap."""

    def read(self, ctx: ObsContext) -> np.ndarray:
        return _require(ctx.explore_map, "explore_map")


def build_local_map(
    tile_visits: set[tuple[int, int, int]],
    map_id: int,
    x: int,
    y: int,
    size: int = LOCAL_MAP_SIZE,
) -> np.ndarray:
    """Render the visited tiles around ``(map_id, x, y)`` at one tile per cell.

    The player is always at the center, so the policy does not have to learn
    where a global marker sits inside a 48x48 image. Visits are keyed by map,
    so another map's tiles cannot leak into the window.
    """
    center = size // 2
    view = np.zeros((size, size), dtype=np.uint8)
    for row in range(size):
        world_y = y + row - center
        for col in range(size):
            world_x = x + col - center
            if (map_id, world_x, world_y) in tile_visits:
                view[row, col] = 255
    view[center, center] = LOCAL_MAP_PLAYER
    return view


class LocalMapProducer:
    """15x15 visited-tile window centered on the player's current tile."""

    key = "local_map"

    def space(self) -> spaces.Space[Any]:
        return spaces.Box(
            low=0,
            high=255,
            shape=(LOCAL_MAP_SIZE, LOCAL_MAP_SIZE),
            dtype=np.uint8,
        )

    def reset(self) -> None:
        """No cached state; the environment owns the window."""

    def read(self, ctx: ObsContext) -> np.ndarray:
        return _require(ctx.local_map, "local_map")


class RecentActionsProducer:
    """The last ``ACTION_HISTORY_LEN`` action indices."""

    key = "recent_actions"

    def space(self) -> spaces.Space[Any]:
        return spaces.MultiDiscrete([8] * ACTION_HISTORY_LEN)

    def reset(self) -> None:
        """No cached state; the environment owns the action history."""

    def read(self, ctx: ObsContext) -> np.ndarray:
        return _require(ctx.recent_actions, "recent_actions")
