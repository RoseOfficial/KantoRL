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


def test_h40_tier_matches_levelmax_except_the_horizon() -> None:
    base = load_tier("t2_levelmax").to_config("pokemon_red.gb")
    h40 = load_tier("t2_levelmax_h40").to_config("pokemon_red.gb")

    assert h40.max_steps == 40960
    assert h40.stagnation_steps == 20480
    assert h40.level_source == "max"
    assert h40.level_weight == 4.0
    assert h40.obs_components == base.obs_components


def test_lr1e4_tier_overrides_the_saved_rate_on_resume() -> None:
    config = load_tier("t2_levelmax_lr1e4").to_config("pokemon_red.gb")

    assert config.learning_rate == 1e-4
    assert config.resume_learning_rate == 1e-4
    assert config.level_source == "max"
    assert config.max_steps == 20480


def test_h40_lr1e4_tier_combines_the_horizon_and_rate() -> None:
    config = load_tier("t2_levelmax_h40_lr1e4").to_config("pokemon_red.gb")

    assert config.max_steps == 40960
    assert config.stagnation_steps == 20480
    assert config.learning_rate == 1e-4
    assert config.resume_learning_rate == 1e-4
    assert config.level_source == "max"


def test_h50_lr1e4_tier_combines_the_longer_horizon_and_rate() -> None:
    config = load_tier("t2_levelmax_h50_lr1e4").to_config("pokemon_red.gb")

    assert config.max_steps == 51200
    assert config.stagnation_steps == 25600
    assert config.learning_rate == 1e-4
    assert config.resume_learning_rate == 1e-4


def test_h60_lr5e5_tier_combines_the_longest_horizon_and_slowest_rate() -> None:
    config = load_tier("t2_levelmax_h60_lr5e5").to_config("pokemon_red.gb")

    assert config.max_steps == 61440
    assert config.stagnation_steps == 30720
    assert config.learning_rate == 5e-5
    assert config.resume_learning_rate == 5e-5


def test_local_map_tier_adds_only_the_local_map() -> None:
    base = load_tier("t2_levelmax_h60_lr5e5").config
    local = load_tier("t2_levelmax_h60_lr5e5_lmap").config

    assert local["obs_components"] == [*base["obs_components"], "local_map"]
    for key, value in base.items():
        if key != "obs_components":
            assert local[key] == value


def test_reward_balance_tier_changes_only_reward_and_rate() -> None:
    base = load_tier("t2_levelmax_h60_lr5e5").config
    rew = load_tier("t2_levelmax_h60_lr2e5_rew").config

    assert rew["obs_components"] == base["obs_components"]
    assert rew["level_weight"] == 1.0
    assert rew["explore_weight"] == 1.0
    assert rew["learning_rate"] == 2e-5
    assert rew["resume_learning_rate"] == 2e-5
    changed = {
        "level_weight",
        "explore_weight",
        "learning_rate",
        "resume_learning_rate",
    }
    for key, value in base.items():
        if key not in changed:
            assert rew[key] == value


def test_gamma_probe_changes_only_gamma_and_rate() -> None:
    base = load_tier("t2_levelmax_h60_lr5e5").config
    gamma = load_tier("t2_levelmax_h60_lr2e5_g999").config

    assert gamma["obs_components"] == base["obs_components"]
    assert gamma["gamma"] == 0.999
    assert gamma["learning_rate"] == 2e-5
    assert gamma["resume_learning_rate"] == 2e-5
    changed = {"gamma", "learning_rate", "resume_learning_rate"}
    for key, value in base.items():
        if key not in changed:
            assert gamma[key] == value


def test_marker_tier_changes_only_the_marker_and_rate() -> None:
    base = load_tier("t2_levelmax_h60_lr5e5").config
    marker = load_tier("t2_levelmax_h60_marker_lr2e5").config

    assert marker["map_player_marker"] is True
    assert marker["obs_components"] == base["obs_components"]
    assert marker["gamma"] == base["gamma"]
    assert marker["level_weight"] == base["level_weight"]
    assert marker["explore_weight"] == base["explore_weight"]
    assert marker["learning_rate"] == 2e-5
    assert marker["resume_learning_rate"] == 2e-5
    changed = {"map_player_marker", "learning_rate", "resume_learning_rate"}
    for key, value in base.items():
        if key not in changed:
            assert marker[key] == value


def test_tiles_tier_adds_only_the_local_tiles_and_rate() -> None:
    base = load_tier("t2_levelmax_h60_lr5e5").config
    tiles = load_tier("t2_levelmax_h60_tiles_lr2e5").config

    assert tiles["obs_components"] == [*base["obs_components"], "local_tiles"]
    assert tiles["gamma"] == base["gamma"]
    assert tiles["level_weight"] == base["level_weight"]
    assert tiles["explore_weight"] == base["explore_weight"]
    assert tiles["learning_rate"] == 2e-5
    assert tiles["resume_learning_rate"] == 2e-5
    changed = {"obs_components", "learning_rate", "resume_learning_rate"}
    for key, value in base.items():
        if key not in changed:
            assert tiles[key] == value


def test_clean_tier_is_the_final_ruleset() -> None:
    config = load_tier("t2_levelmax_h60_lr1e4to5e5").to_config("pokemon_red.gb")
    assert config.obs_components == (
        "screens",
        "health",
        "level",
        "badges",
        "events",
        "map",
        "recent_actions",
    )
    assert config.level_weight == 4.0
    assert config.explore_weight == 0.3
    assert config.gamma == 0.995
    assert config.max_steps == 61440
    assert config.learning_rate == 1e-4
    assert config.lr_decay_step == 78_000_000
    assert config.lr_decay_to == 5e-5


def test_phase0_arms_change_one_axis_each() -> None:
    """The Phase 0 screen is only interpretable if each arm is one change."""
    base = load_tier("phase0_baseline").config

    for name, key, value in (
        ("phase0_gamma999", "gamma", 0.999),
        ("phase0_explore3", "explore_weight", 3.0),
        ("phase0_level1", "level_weight", 1.0),
    ):
        arm = load_tier(name).config
        changed = {k for k in set(base) | set(arm) if base.get(k) != arm.get(k)}
        assert changed == {key}, f"{name} changed {sorted(changed)}"
        assert arm[key] == value


def test_phase1_tier_is_recurrent_archive_without_hm_automation() -> None:
    config = load_tier("phase1_recurrent_tiles").to_config("pokemon_red.gb")

    assert config.recurrent is True
    assert config.enable_curriculum is True
    assert config.enable_hm_automation is False
    assert config.dynamic_episode_length is False
    assert "local_tiles" in config.obs_components
    assert config.explore_weight == 3.0
    assert config.gamma == 0.995


def test_phase1b_tier_fixes_the_archive_weights() -> None:
    config = load_tier("phase1b_recurrent_tiles").to_config("pokemon_red.gb")

    assert config.checkpoint_weight == 0.25
    assert config.archive_frontier_fraction == 0.8
    assert config.archive_recent_fraction == 0.2
    assert config.archive_require_progress is True
    assert config.recurrent is True


def test_phase2_arms_isolate_archive_and_tiles() -> None:
    control = load_tier("phase2_control").config
    tiles = load_tier("phase2_tiles").config
    archive = load_tier("phase2_archive50").config
    both = load_tier("phase2_tiles_archive50").config

    assert control["enable_curriculum"] is False
    assert control["recurrent"] is False
    assert "local_tiles" not in control["obs_components"]

    assert set(tiles["obs_components"]) == set(control["obs_components"]) | {
        "local_tiles"
    }
    assert tiles["enable_curriculum"] is False

    assert archive["obs_components"] == control["obs_components"]
    assert archive["enable_curriculum"] is True
    assert archive["checkpoint_weight"] == 0.5
    assert archive["recurrent"] is False

    assert set(both["obs_components"]) == set(tiles["obs_components"])
    assert both["enable_curriculum"] is True

    for arm in (tiles, archive, both):
        for key in (
            "explore_weight",
            "level_weight",
            "gamma",
            "learning_rate",
            "max_steps",
            "n_envs",
        ):
            assert arm[key] == control[key]


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
