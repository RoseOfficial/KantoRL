"""Tests for training-configuration precedence.

The rule these tests protect: a tier file defines the experiment, the CLI
supplies machine-specific paths and can only *add* features, and neither can
silently change the other. Getting this wrong means a run is not the run you
think it is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kantorl.config import KantoConfig
from kantorl.obs import DEFAULT_OBS_COMPONENTS
from kantorl.tiers import load_tier
from kantorl.train import resolve_config


def resolve(base_config=None, **overrides):
    """Call resolve_config with neutral defaults for everything not under test."""
    kwargs = {
        "rom_path": "pokemon_red.gb",
        "session_path": "runs/test",
        "n_envs": None,
        "enable_streaming": False,
        "stream_username": "KantoRL",
        "stream_color": "#ff0000",
        "stream_sprite_id": 0,
        "use_curriculum": False,
        "enable_wandb": False,
        "wandb_project": "kantorl",
        "wandb_entity": None,
        "wandb_group": None,
        "use_agent": False,
        "obs_components": None,
        "base_config": base_config,
    }
    kwargs.update(overrides)
    return resolve_config(**kwargs)


# =============================================================================
# PLAIN RUN (NO TIER)
# =============================================================================


def test_plain_run_keeps_legacy_defaults() -> None:
    config = resolve()

    assert config.rom_path == "pokemon_red.gb"
    assert config.obs_components == DEFAULT_OBS_COMPONENTS
    assert config.n_envs == 16
    assert config.enable_curriculum is False


def test_plain_run_honours_observed_components() -> None:
    config = resolve(obs_components=("screens",))
    assert config.obs_components == ("screens",)


def test_plain_run_n_envs_can_be_set() -> None:
    assert resolve(n_envs=4).n_envs == 4


# =============================================================================
# TIER RUN
# =============================================================================


def test_tier_supplies_environment_and_ppo_settings() -> None:
    tier = load_tier("t0_pure")
    config = resolve(base_config=tier.to_config("old.gb"))

    assert config.obs_components == ("screens",)
    assert config.reward_scale == tier.config["reward_scale"]
    assert config.gamma == tier.config["gamma"]
    assert config.max_steps == tier.config["max_steps"]
    assert config.n_envs == tier.config["n_envs"]


def test_tier_supplies_the_kl_guard_and_run_scoped_interactions() -> None:
    tier = load_tier("t2_pragmatic")
    config = resolve(base_config=tier.to_config("old.gb"))

    assert config.target_kl == 0.02
    assert config.interact_once_per_run is True


def test_target_kl_must_be_positive_or_none() -> None:
    with pytest.raises(ValueError, match="target_kl"):
        KantoConfig(rom_path="pokemon_red.gb", target_kl=0)

    assert KantoConfig(rom_path="pokemon_red.gb", target_kl=None).target_kl is None


def test_level_source_must_be_sum_or_max() -> None:
    with pytest.raises(ValueError, match="level_source"):
        KantoConfig(rom_path="pokemon_red.gb", level_source="average")


def test_resume_learning_rate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="resume_learning_rate"):
        KantoConfig(rom_path="pokemon_red.gb", resume_learning_rate=0)


def test_paths_always_come_from_the_caller() -> None:
    """A tier file cannot know where the ROM lives; the caller wins."""
    config = resolve(
        base_config=load_tier("t1_minimal").to_config("stale.gb", "runs/stale"),
        rom_path="real.gb",
        session_path="runs/real",
    )

    assert config.rom_path == "real.gb"
    assert Path(config.session_path) == Path("runs/real")


def test_cli_obs_override_wins_over_tier() -> None:
    config = resolve(
        base_config=load_tier("t2_pragmatic").to_config("pokemon_red.gb"),
        obs_components=("screens",),
    )
    assert config.obs_components == ("screens",)


def test_cli_n_envs_override_wins_over_tier() -> None:
    tier = load_tier("t1_minimal").to_config("pokemon_red.gb")
    assert resolve(base_config=tier, n_envs=2).n_envs == 2


def test_tier_feature_stays_on_when_cli_is_silent() -> None:
    """The CLI's False default must not disable a feature the tier enabled."""
    tier = load_tier("t1_minimal").to_config("pokemon_red.gb")
    tier.enable_curriculum = True

    config = resolve(base_config=tier, use_curriculum=False)
    assert config.enable_curriculum is True


def test_cli_can_add_a_feature_the_tier_left_off() -> None:
    tier = load_tier("t0_pure").to_config("pokemon_red.gb")
    assert tier.enable_agent is False

    config = resolve(base_config=tier, use_agent=True)
    assert config.enable_agent is True


def test_expanded_tier_config_is_validated() -> None:
    """A resolved tier config must still be constructible and self-consistent."""
    config = resolve(base_config=load_tier("t2_pragmatic").to_config("pokemon_red.gb"))
    assert isinstance(config, KantoConfig)
    assert config.frame_stacks >= 1
    assert 0 < config.reward_scale <= 10
