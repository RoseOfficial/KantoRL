"""Named training tiers.

A tier is a complete experiment definition: which observation components the
policy sees, how rewards and horizons are shaped, and whether curriculum or the
agent module is enabled. The tiers are ordered by how much hand-authored game
knowledge they assume, not by how well they are expected to score.

The point of naming them is comparability. Two runs are only a controlled
comparison if everything except the ablated axis is identical, so the tier
files hold the reward, horizon and environment settings fixed and let the
observation differ.

Files live next to this module so they travel with the package::

    kantorl tier list
    kantorl tier run t1_minimal pokemon_red.gb --session runs/t1

A tier file has two sections: ``config`` keys map directly onto
:class:`kantorl.config.KantoConfig` fields, and ``train`` keys map onto the
training loop arguments. Unknown keys are rejected rather than ignored, because
a typo in an experiment definition silently changes what you measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from kantorl.config import KantoConfig

__all__ = [
    "BUILTIN_TIER_NAMES",
    "Tier",
    "available_tier_names",
    "load_tier",
]

# Bundled tiers, ordered from fewest priors to most.
#
# The first three are the canonical ladder: they differ only in observation.
# The next two are single-variable ablation arms used to locate *which*
# observation carries an effect, not rungs of that ladder. t2_level4,
# t2_levelmax, t2_levelmax_h40, t2_levelmax_lr1e4, t2_levelmax_h40_lr1e4,
# t2_levelmax_h50_lr1e4 and t2_levelmax_h60_lr5e5 are reward/horizon/optimizer
# variants: same observation as t2_pragmatic. t2_levelmax_h60_lr5e5_lmap adds
# the opt-in 15x15 local visited-tile window; t2_levelmax_h60_lr2e5_rew keeps
# the legacy observation and changes only the reward balance and rate;
# t2_levelmax_h60_lr2e5_g999 changes only gamma and the rate.
BUILTIN_TIER_NAMES: tuple[str, ...] = (
    "t0_pure",
    "t1_minimal",
    "t2_pragmatic",
    "t2_screens_map",
    "t2_all_no_map",
    "t2_level4",
    "t2_levelmax",
    "t2_levelmax_h40",
    "t2_levelmax_lr1e4",
    "t2_levelmax_h40_lr1e4",
    "t2_levelmax_h50_lr1e4",
    "t2_levelmax_h60_lr5e5",
    "t2_levelmax_h60_lr5e5_lmap",
    "t2_levelmax_h60_lr2e5_rew",
    "t2_levelmax_h60_lr2e5_g999",
)

# Keys allowed in the ``train`` section. These are loop-level arguments, not
# environment configuration.
_TRAIN_KEYS = frozenset({"total_timesteps", "reward_fn", "seed", "resume"})


@dataclass(frozen=True)
class Tier:
    """A named experiment definition.

    Attributes:
        name: Tier identifier, e.g. ``"t1_minimal"``.
        description: Why this tier exists and what it ablates.
        config: Overrides applied to :class:`KantoConfig`.
        train: Training loop arguments (``total_timesteps``, ``reward_fn``, ...).
    """

    name: str
    description: str
    config: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)

    def to_config(
        self,
        rom_path: str | Path,
        session_path: str | Path | None = None,
    ) -> KantoConfig:
        """Build the environment configuration for this tier.

        Args:
            rom_path: Pokemon Red ROM path.
            session_path: Optional session directory override.

        Returns:
            A validated :class:`KantoConfig`.
        """
        overrides = dict(self.config)
        overrides["rom_path"] = str(rom_path)
        if session_path is not None:
            overrides["session_path"] = str(session_path)
        return KantoConfig.from_dict(overrides)

    def train_kwargs(self) -> dict[str, Any]:
        """Return the training loop arguments for this tier."""
        return dict(self.train)


def _parse_tier(raw: Any, source: str) -> Tier:
    """Validate and build a tier from decoded YAML.

    Raises:
        ValueError: If the structure is wrong or a key is unknown.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Tier file {source} must contain a mapping at the top level")

    name = str(raw.get("name") or Path(source).stem)
    description = str(raw.get("description") or "").strip()

    config_raw = raw.get("config") or {}
    train_raw = raw.get("train") or {}
    if not isinstance(config_raw, dict):
        raise ValueError(f"Tier {name}: 'config' must be a mapping")
    if not isinstance(train_raw, dict):
        raise ValueError(f"Tier {name}: 'train' must be a mapping")

    valid_config = set(KantoConfig.__dataclass_fields__)
    unknown_config = sorted(set(config_raw) - valid_config)
    if unknown_config:
        raise ValueError(
            f"Tier {name}: unknown config key(s) {unknown_config}. "
            f"Valid keys are KantoConfig fields."
        )

    unknown_train = sorted(set(train_raw) - _TRAIN_KEYS)
    if unknown_train:
        raise ValueError(
            f"Tier {name}: unknown train key(s) {unknown_train}. "
            f"Allowed: {sorted(_TRAIN_KEYS)}"
        )

    # Construct once so bad values fail at load time, not hours into training.
    KantoConfig.from_dict({**config_raw, "rom_path": "validation.gb"})

    return Tier(
        name=name,
        description=description,
        config=dict(config_raw),
        train=dict(train_raw),
    )


def available_tier_names() -> tuple[str, ...]:
    """Names of the bundled tiers, ordered from fewest to most priors."""
    return BUILTIN_TIER_NAMES


def _read_builtin(name: str) -> str:
    """Read a bundled tier YAML by name."""
    if name not in BUILTIN_TIER_NAMES:
        raise ValueError(
            f"Unknown tier {name!r}. Available: {list(BUILTIN_TIER_NAMES)}"
        )
    resource = resources.files("kantorl.tiers").joinpath(f"{name}.yaml")
    return resource.read_text(encoding="utf-8")


def load_tier(spec: str | Path) -> Tier:
    """Load a tier from a name or a YAML path.

    Args:
        spec: A bundled tier name (``"t1_minimal"``) or a path to a YAML file.

    Returns:
        The parsed tier.

    Raises:
        FileNotFoundError: If a path is given and does not exist.
        ValueError: If the tier is unknown or malformed.
    """
    path = Path(spec)
    if path.is_file():
        return _parse_tier(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))

    return _parse_tier(yaml.safe_load(_read_builtin(str(spec))), str(spec))
