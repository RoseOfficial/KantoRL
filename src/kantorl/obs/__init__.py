"""Composable observation package.

``KantoConfig.obs_components`` selects which producers are active. The default
tuple reproduces the historical observation exactly, so existing checkpoints
and training runs are unaffected. Dropping components is how the observation
tiers are expressed:

* ``("screens",)`` — pixels only; closest to tabula rasa.
* ``("screens", "health", "events")`` — minimal progress signal.
* the default seven — full RAM-rich observation.

Adding a new observation component means writing one producer and registering
it here; the environment does not need to change.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kantorl.obs.base import ObsContext, ObsProducer
from kantorl.obs.producers import (
    ACTION_HISTORY_LEN,
    EVENT_OBS_DIM,
    LEVEL_OBS_DIM,
    MAP_OBS_SHAPE,
    BadgesProducer,
    EventsProducer,
    HealthProducer,
    LevelProducer,
    MapProducer,
    RecentActionsProducer,
    ScreenProducer,
)

# Legacy observation, in its original dict order.
DEFAULT_OBS_COMPONENTS: tuple[str, ...] = (
    "screens",
    "health",
    "level",
    "badges",
    "events",
    "map",
    "recent_actions",
)

# Factories take the config and return a producer. They are intentionally
# lambdas rather than classes so each producer's constructor stays explicit.
_FACTORIES: dict[str, Callable[[Any], ObsProducer]] = {
    "screens": lambda cfg: ScreenProducer(cfg.frame_stacks, cfg.screen_size),
    "health": lambda cfg: HealthProducer(),
    "level": lambda cfg: LevelProducer(),
    "badges": lambda cfg: BadgesProducer(),
    "events": lambda cfg: EventsProducer(),
    "map": lambda cfg: MapProducer(),
    "recent_actions": lambda cfg: RecentActionsProducer(),
}

# Every component name, in the order used by the default observation.
OBS_COMPONENTS: tuple[str, ...] = tuple(_FACTORIES)

if set(DEFAULT_OBS_COMPONENTS) != set(OBS_COMPONENTS):
    raise RuntimeError("DEFAULT_OBS_COMPONENTS and producer registry have drifted")


def build_producers(config: Any) -> list[ObsProducer]:
    """Instantiate the producers selected by ``config.obs_components``.

    Args:
        config: A :class:`kantorl.config.KantoConfig`-like object. Only
            ``obs_components``, ``frame_stacks``, and ``screen_size`` are read.

    Returns:
        Producers in config order, ready to be asked for spaces and reads.

    Raises:
        ValueError: If the config names an unknown component.
    """
    names = tuple(config.obs_components)
    unknown = [name for name in names if name not in _FACTORIES]
    if unknown:
        raise ValueError(
            f"Unknown observation component(s): {unknown}. "
            f"Available: {sorted(_FACTORIES)}"
        )
    return [_FACTORIES[name](config) for name in names]


__all__ = [
    "ACTION_HISTORY_LEN",
    "DEFAULT_OBS_COMPONENTS",
    "EVENT_OBS_DIM",
    "LEVEL_OBS_DIM",
    "MAP_OBS_SHAPE",
    "OBS_COMPONENTS",
    "ObsContext",
    "ObsProducer",
    "build_producers",
]
