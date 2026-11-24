"""Core types for composable observations.

The environment used to hard-code one observation dict. It now composes that
dict from a list of :class:`ObsProducer` instances selected by
``KantoConfig.obs_components``. Each producer owns one key, its Gymnasium
space, and the logic that reads it from a shared :class:`ObsContext`.

Why a shared context?
    Several producers read the same memory (HP, event flags, badges). The
    environment reads those values once per step and puts them in the context,
    so adding an observation component does not add redundant memory traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from pyboy import PyBoy


@dataclass(frozen=True)
class ObsContext:
    """Per-step snapshot shared by all active observation producers.

    Fields that are not observed in the current tier may be ``None``; the
    corresponding producer is simply not constructed. ``levels`` and the HP
    fields are always populated because they are cheap to read and useful for
    logging, even when they are not part of the policy input.

    Attributes:
        pyboy: Live emulator instance, for producers that need extra memory.
        frame_stack: ``(frame_stacks, height, width)`` grayscale frames.
        explore_map: ``(48, 48)`` bitmap of visited global coordinates.
        local_map: ``(15, 15)`` visited-tile window centered on the player.
        recent_actions: Last ``ACTION_HISTORY_LEN`` action indices.
        event_flags: ``(EVENT_OBS_DIM,)`` binary event flags.
        badge_flags: ``(8,)`` binary gym badge flags.
        levels: Party Pokemon levels.
        current_hp: Total current HP across the party.
        max_hp: Total maximum HP across the party.
    """

    pyboy: PyBoy
    frame_stack: np.ndarray | None = None
    explore_map: np.ndarray | None = None
    local_map: np.ndarray | None = None
    recent_actions: np.ndarray | None = None
    event_flags: np.ndarray | None = None
    badge_flags: np.ndarray | None = None
    levels: tuple[int, ...] = ()
    current_hp: int = 0
    max_hp: int = 0


class ObsProducer(Protocol):
    """One observation component: a dict key, a space, and a reader.

    Producers are stateless unless they cache something derived (see
    :class:`~kantorl.obs.producers.LevelProducer`). ``reset`` is called at the
    start of every episode so caches cannot leak across resets.
    """

    key: str

    def space(self) -> spaces.Space[Any]:
        """Return the Gymnasium space for this component."""
        ...

    def reset(self) -> None:
        """Clear any episode-local cache."""
        ...

    def read(self, ctx: ObsContext) -> np.ndarray:
        """Read this component from the per-step context."""
        ...
