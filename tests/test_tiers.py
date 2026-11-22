"""Tests for bundled tier definitions.

Tiers are experiment definitions, so the tests focus on the properties that
make an experiment interpretable: they must parse, they must be ablations of
each other along a single axis, and a typo must fail loudly rather than
silently shift the result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kantorl.config import KantoConfig
from kantorl.obs import OBS_COMPONENTS
from kantorl.tiers import BUILTIN_TIER_NAMES, available_tier_names, load_tier


def test_all_builtin_tiers_load() -> None:
    assert available_tier_names() == BUILTIN_TIER_NAMES
    for name in BUILTIN_TIER_NAMES:
        tier = load_tier(name)
        assert tier.name == name
        assert tier.description
        assert tier.config["obs_components"]


def test_tier_obs_components_are_valid() -> None:
    for name in BUILTIN_TIER_NAMES:
        tier = load_tier(name)
        for component in tier.config["obs_components"]:
            assert component in OBS_COMPONENTS


def test_to_config_builds_valid_env_config() -> None:
    tier = load_tier("t0_pure")
    config = tier.to_config("pokemon_red.gb", "runs/t0")

    assert isinstance(config, KantoConfig)
    assert config.rom_path == "pokemon_red.gb"
    assert config.obs_components == ("screens",)
    assert str(config.session_path) == "runs\\t0" or str(config.session_path) == "runs/t0"


def test_t0_is_the_leanest_tier() -> None:
    """T0 must expose pixels and nothing else."""
    t0 = load_tier("t0_pure")
    assert t0.config["obs_components"] == ["screens"]
    assert t0.train["reward_fn"] == "default"


def test_t1_and_t2_share_reward_and_horizon() -> None:
    """Only the observation may differ between T1 and T2, or the comparison lies."""
    t1 = load_tier("t1_minimal").config
    t2 = load_tier("t2_pragmatic").config

    for key in (
        "max_steps",
        "reward_scale",
        "explore_weight",
        "gamma",
        "gae_lambda",
        "frame_stacks",
        "screen_size",
        "action_freq",
        "enable_curriculum",
        "enable_agent",
        "target_kl",
        "interact_once_per_run",
    ):
        if key in t1 or key in t2:
            assert t1.get(key) == t2.get(key), f"{key} differs between T1 and T2"

    assert set(t1["obs_components"]) < set(t2["obs_components"])


def test_all_tiers_share_the_optimizer_guard_and_interaction_fix() -> None:
    """These are harness settings, not observation priors: they must not differ."""
    for name in BUILTIN_TIER_NAMES:
        config = load_tier(name).config
        assert config["target_kl"] == 0.02
        assert config["interact_once_per_run"] is True


def test_levelmax_tier_uses_the_max_level_source() -> None:
    tier = load_tier("t2_levelmax")
    config = tier.to_config("pokemon_red.gb")

    assert config.level_source == "max"
    assert config.level_weight == 4.0
    assert config.obs_components == tuple(tier.config["obs_components"])


def test_t2_uses_pixels_and_more_priors_than_t1() -> None:
    t1 = set(load_tier("t1_minimal").config["obs_components"])
    t2 = set(load_tier("t2_pragmatic").config["obs_components"])

    assert {"screens", "health"} <= t1
    assert {"events", "map", "recent_actions"} <= t2 - t1
    assert "screens" in t2


def write_tier(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "tier.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = write_tier(
        tmp_path,
        "name: bad\nconfig:\n  obs_component: [screens]\n",
    )
    with pytest.raises(ValueError, match="unknown config key"):
        load_tier(path)


def test_unknown_train_key_is_rejected(tmp_path: Path) -> None:
    path = write_tier(
        tmp_path,
        "name: bad\ntrain:\n  total_step: 100\n",
    )
    with pytest.raises(ValueError, match="unknown train key"):
        load_tier(path)


def test_invalid_obs_component_is_rejected(tmp_path: Path) -> None:
    path = write_tier(
        tmp_path,
        "name: bad\nconfig:\n  obs_components: [screens, lidar]\n",
    )
    with pytest.raises(ValueError, match="Unknown observation component"):
        load_tier(path)


def test_unknown_tier_name_lists_alternatives() -> None:
    with pytest.raises(ValueError, match="t0_pure"):
        load_tier("does_not_exist")


def test_tier_with_no_description_defaults_to_empty(tmp_path: Path) -> None:
    path = write_tier(tmp_path, "name: bare\nconfig:\n  obs_components: [screens]\n")
    tier = load_tier(path)
    assert tier.description == ""
    assert tier.train == {}


def test_all_tiers_share_reward_and_horizon() -> None:
    """The observation must be the only axis that differs between tiers."""
    tiers = [load_tier(name) for name in ("t0_pure", "t1_minimal", "t2_pragmatic")]

    assert {t.train["reward_fn"] for t in tiers} == {"default"}
    assert len({(t.config["max_steps"], t.config["stagnation_steps"]) for t in tiers}) == 1
    assert len(
        {
            (
                t.config["reward_scale"],
                t.config["explore_weight"],
                t.config["interact_weight"],
                t.config["level_weight"],
                t.config["stuck_penalty"],
            )
            for t in tiers
        }
    ) == 1
