"""Tests for the modular feature extractor.

The bug this protects: with the stock `MultiInputPolicy` extractor, a trained
T2 policy ignored five of its seven observation components -- randomising the
event vector, badges, levels, HP or recent actions changed the chosen action on
0% of 250 transitions. These tests check the structure that is supposed to fix
that: a dedicated encoder per key, a fixed-width fusion, and no component
silently flattened into a shared MLP.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from kantorl.encoders import ModularExtractor
from kantorl.train import create_policy_kwargs


def t2_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "screens": spaces.Box(0, 255, (3, 72, 80), np.uint8),
            "health": spaces.Box(0.0, 1.0, (1,), np.float32),
            "level": spaces.Box(-1.0, 1.0, (8,), np.float32),
            "badges": spaces.MultiBinary(8),
            "events": spaces.MultiBinary(2560),
            "map": spaces.Box(0, 255, (48, 48), np.uint8),
            "recent_actions": spaces.MultiDiscrete([8, 8, 8]),
        }
    )


def batch(space: spaces.Dict, n: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    out = {}
    for key, sub in space.spaces.items():
        if isinstance(sub, spaces.MultiBinary):
            out[key] = torch.as_tensor(rng.integers(0, 2, (n, *sub.shape)))
        elif isinstance(sub, spaces.MultiDiscrete):
            out[key] = torch.as_tensor(
                np.stack([rng.integers(0, int(k), n) for k in sub.nvec], axis=1)
            )
        elif len(sub.shape) == 3:
            out[key] = torch.as_tensor(rng.integers(0, 256, (n, *sub.shape), dtype=np.uint8))
        elif len(sub.shape) == 2 and sub.dtype == np.uint8:
            out[key] = torch.as_tensor(rng.integers(0, 256, (n, *sub.shape), dtype=np.uint8))
        else:
            out[key] = torch.as_tensor(rng.random((n, *sub.shape), dtype=np.float32))
    return out


def test_every_component_gets_its_own_encoder():
    space = t2_space()
    extractor = ModularExtractor(space)

    encoded = (
        set(extractor.vector_encoders)
        | set(extractor.image_encoders)
        | set(extractor.tile_encoders)
    )

    assert encoded == set(space.spaces)
    assert "screens" in extractor.image_encoders


def test_tile_maps_get_an_embedding_encoder():
    """Raw tile IDs are categorical; they must not be flattened as scalars."""
    space = spaces.Dict(
        {
            "screens": spaces.Box(0, 255, (3, 72, 80), np.uint8),
            "local_tiles": spaces.Box(0, 255, (15, 15), np.uint8),
        }
    )
    extractor = ModularExtractor(space)

    assert "local_tiles" in extractor.tile_encoders
    out = extractor(batch(space))
    assert out.shape == (4, 512)
    assert torch.isfinite(out).all()


def test_output_width_is_fixed_regardless_of_component_count():
    """The policy must see the same capacity in every tier."""
    t0 = spaces.Dict({"screens": spaces.Box(0, 255, (3, 72, 80), np.uint8)})
    t2 = t2_space()

    out0 = ModularExtractor(t0)(batch(t0))
    out2 = ModularExtractor(t2)(batch(t2))

    assert out0.shape == out2.shape == (4, 512)


def test_changing_an_auxiliary_component_changes_the_features():
    """No component may be a no-op pass-through into the fused vector."""
    space = t2_space()
    extractor = ModularExtractor(space)
    base = batch(space)

    with torch.no_grad():
        reference = extractor(base)
        for key in ("events", "badges", "level", "health", "recent_actions"):
            perturbed = {k: v.clone() for k, v in base.items()}
            if isinstance(space[key], spaces.MultiBinary):
                perturbed[key] = 1 - perturbed[key]
            elif isinstance(space[key], spaces.MultiDiscrete):
                perturbed[key] = (perturbed[key] + 1) % space[key].nvec
            else:
                perturbed[key] = 1.0 - perturbed[key]
            delta = (extractor(perturbed) - reference).abs().max().item()
            assert delta > 0, f"{key} does not affect the fused features"


def test_multidiscrete_is_one_hot_encoded_not_treated_as_a_scalar():
    space = spaces.Dict({"recent_actions": spaces.MultiDiscrete([8, 8, 8])})
    extractor = ModularExtractor(space)

    a = torch.tensor([[0, 0, 0]])
    b = torch.tensor([[7, 7, 7]])
    with torch.no_grad():
        assert not torch.allclose(extractor({"recent_actions": a}),
                                  extractor({"recent_actions": b}))


def test_single_component_space_still_works():
    space = spaces.Dict({"health": spaces.Box(0.0, 1.0, (1,), np.float32)})
    out = ModularExtractor(space)(batch(space))
    assert out.shape == (4, 512)
    assert torch.isfinite(out).all()


def test_create_policy_kwargs_uses_the_modular_extractor_by_default():
    from kantorl.train import create_policy_kwargs

    kwargs = create_policy_kwargs()
    assert kwargs["features_extractor_class"] is ModularExtractor
    assert create_policy_kwargs(modular=False).get("features_extractor_class") is None


def test_policy_builds_and_predicts_with_the_modular_extractor():
    """End-to-end shape check through SB3, without touching an emulator."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    space = t2_space()

    class FakeEnv(gym.Env):
        observation_space = space
        action_space = spaces.Discrete(8)

        def __init__(self) -> None:
            super().__init__()
            self._n = 0

        def reset(self):
            self._n += 1
            return batch(space, n=1, seed=self._n)

        def step(self, action):
            self._n += 1
            obs = batch(space, n=1, seed=self._n)
            return obs, 0.0, False, False, {}

    model = PPO(
        "MultiInputPolicy",
        DummyVecEnv([FakeEnv]),
        n_steps=8,
        batch_size=8,
        policy_kwargs=create_policy_kwargs(),
        device="cpu",
    )
    action, _ = model.predict(batch(space, n=3), deterministic=True)
    assert action.shape == (3,)
