"""Load Stable-Baselines3 checkpoints and recover their environment contract.

A checkpoint is only meaningful next to the environment it was trained on: the
observation tier, frame stack and screen size all change tensor shapes, and a
recurrent policy needs an LSTM state threaded through `predict`. This module
answers those questions by reading the checkpoint's own metadata, so evaluators
do not have to guess or ask the user.

The metadata read is pure ``zipfile`` + ``json``; torch is only imported when a
model is actually loaded.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from gymnasium import spaces

from kantorl.obs import DEFAULT_OBS_COMPONENTS, OBS_COMPONENTS

__all__ = [
    "assert_observation_compatible",
    "env_kwargs_from_model",
    "is_recurrent",
    "is_recurrent_checkpoint",
    "load_model",
    "observation_components",
    "read_checkpoint_metadata",
]


def read_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read the JSON ``data`` entry that Stable-Baselines3 writes into a zip.

    Args:
        checkpoint_path: Path to a ``.zip`` checkpoint saved by ``model.save()``.

    Returns:
        The decoded metadata dict. Empty if the entry is missing or malformed.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the file is not a Stable-Baselines3 checkpoint.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with zipfile.ZipFile(path) as archive:
        try:
            raw = archive.read("data")
        except KeyError as exc:
            raise ValueError(
                f"{path} is not a Stable-Baselines3 checkpoint "
                "(no 'data' entry in the archive)."
            ) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # A corrupt data entry should not block loading weights; the policy
        # class check is a convenience, not a requirement.
        return {}


def is_recurrent_checkpoint(checkpoint_path: str | Path) -> bool:
    """Whether a checkpoint was saved with ``sb3_contrib.RecurrentPPO``.

    Detection is by policy class name, which Stable-Baselines3 records in its
    metadata as a fully qualified string such as
    ``sb3_contrib.common.recurrent.policies.RecurrentMultiInputActorCriticPolicy``.
    """
    policy_class = str(read_checkpoint_metadata(checkpoint_path).get("policy_class", ""))
    return "recurrent" in policy_class.lower()


def is_recurrent(model: Any) -> bool:
    """Whether a loaded model expects an LSTM state passed to ``predict``."""
    return type(model).__name__ == "RecurrentPPO"


def load_model(checkpoint_path: str | Path, device: str = "auto") -> Any:
    """Load a PPO or RecurrentPPO checkpoint, choosing the class from metadata.

    Args:
        checkpoint_path: Path to the ``.zip`` checkpoint.
        device: Torch device string passed through to Stable-Baselines3.

    Returns:
        The loaded model.

    Raises:
        ImportError: If the checkpoint is recurrent and ``sb3-contrib`` is
            not installed.
    """
    path = str(checkpoint_path)

    if is_recurrent_checkpoint(path):
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "This checkpoint uses a recurrent (LSTM) policy. "
                "Install sb3-contrib to evaluate it:\n"
                "    pip install sb3-contrib"
            ) from exc
        return RecurrentPPO.load(path, device=device)

    from stable_baselines3 import PPO

    return PPO.load(path, device=device)


def observation_components(model: Any) -> tuple[str, ...]:
    """Reconstruct the observation tier a model was trained on.

    Checkpoints record their observation space, so a pixels-only model can be
    evaluated without being told it is pixels-only. Unknown keys are dropped;
    a non-dict space falls back to the legacy seven-component observation.
    """
    space = getattr(model, "observation_space", None)
    if not isinstance(space, spaces.Dict):
        return DEFAULT_OBS_COMPONENTS

    known = tuple(key for key in space.spaces if key in OBS_COMPONENTS)
    return known or DEFAULT_OBS_COMPONENTS


def env_kwargs_from_model(model: Any) -> dict[str, Any]:
    """Config kwargs needed to rebuild the environment a model was trained in.

    Returns:
        A dict suitable for ``KantoConfig(**kwargs)`` containing
        ``obs_components`` and, when the checkpoint observes screens, the
        ``frame_stacks`` and ``screen_size`` implied by the image space.
    """
    space = getattr(model, "observation_space", None)
    kwargs: dict[str, Any] = {"obs_components": observation_components(model)}

    if not isinstance(space, spaces.Dict):
        return kwargs

    screens = space.spaces.get("screens")
    if isinstance(screens, spaces.Box) and len(screens.shape) == 3:
        channels, height, width = screens.shape
        kwargs["frame_stacks"] = int(channels)
        kwargs["screen_size"] = (int(height), int(width))

    return kwargs


def assert_observation_compatible(model: Any, env: Any) -> None:
    """Fail loudly when a checkpoint cannot read an environment's observations.

    Stable-Baselines3's error for a shape mismatch is a raw stack trace from
    inside the policy. This check turns the common cases (wrong tier, agent
    augmentation, different frame stack) into a message that says what to do.

    Raises:
        ValueError: If the dict keys or any component shape differ.
    """
    model_space = getattr(model, "observation_space", None)
    env_space = getattr(env, "observation_space", None)

    if not isinstance(model_space, spaces.Dict) or not isinstance(env_space, spaces.Dict):
        return

    model_keys = set(model_space.spaces)
    env_keys = set(env_space.spaces)
    if model_keys != env_keys:
        raise ValueError(
            "Checkpoint observation keys do not match the environment: "
            f"checkpoint={sorted(model_keys)}, env={sorted(env_keys)}. "
            "The configuration used for evaluation must expose the same "
            "components the policy was trained with."
        )

    for key in sorted(model_keys):
        expected = model_space.spaces[key]
        actual = env_space.spaces[key]
        if expected.shape != actual.shape:
            raise ValueError(
                f"Observation '{key}' shape mismatch: checkpoint expects "
                f"{expected.shape}, environment provides {actual.shape}. "
                "Frame stack or screen size differs from training."
            )
