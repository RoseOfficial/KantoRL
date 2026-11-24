"""Tests for the composable observation layer.

These tests do not need a ROM: producers only read the ObsContext, and the
environment builds its spaces without touching PyBoy (the emulator is lazily
initialized).
"""

from __future__ import annotations

import numpy as np
import pytest

from kantorl.config import KantoConfig
from kantorl.env import KantoRedEnv
from kantorl.obs import (
    ACTION_HISTORY_LEN,
    DEFAULT_OBS_COMPONENTS,
    EVENT_OBS_DIM,
    LOCAL_MAP_SIZE,
    MAP_OBS_SHAPE,
    OBS_COMPONENTS,
    ObsContext,
    build_producers,
)
from kantorl.obs.producers import (
    EventsProducer,
    HealthProducer,
    LevelProducer,
    LocalMapProducer,
    MapProducer,
    RecentActionsProducer,
    ScreenProducer,
    build_local_map,
)


def make_context(**overrides: object) -> ObsContext:
    """Build a minimal context, overriding only the fields a test needs."""
    defaults: dict[str, object] = {
        "pyboy": None,
        "frame_stack": np.zeros((3, 72, 80), dtype=np.uint8),
        "explore_map": np.zeros(MAP_OBS_SHAPE, dtype=np.uint8),
        "recent_actions": np.zeros(ACTION_HISTORY_LEN, dtype=np.int64),
        "event_flags": np.zeros(EVENT_OBS_DIM, dtype=np.int8),
        "badge_flags": np.zeros(8, dtype=np.int8),
        "levels": (),
        "current_hp": 0,
        "max_hp": 0,
    }
    defaults.update(overrides)
    return ObsContext(**defaults)  # type: ignore[arg-type]


# =============================================================================
# REGISTRY AND CONFIG
# =============================================================================


def test_default_components_match_registry() -> None:
    """The legacy default is a subset; the local map is opt-in."""
    assert set(DEFAULT_OBS_COMPONENTS).issubset(OBS_COMPONENTS)
    assert set(DEFAULT_OBS_COMPONENTS) == {
        "screens",
        "health",
        "level",
        "badges",
        "events",
        "map",
        "recent_actions",
    }
    assert "local_map" in OBS_COMPONENTS
    assert "local_map" not in DEFAULT_OBS_COMPONENTS


def test_build_producers_preserves_config_order() -> None:
    config = KantoConfig(
        rom_path="fake.gb",
        obs_components=("recent_actions", "screens"),
    )
    producers = build_producers(config)
    assert [p.key for p in producers] == ["recent_actions", "screens"]


def test_unknown_component_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown observation component"):
        KantoConfig(rom_path="fake.gb", obs_components=("screens", "lidar"))


def test_duplicate_component_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        KantoConfig(rom_path="fake.gb", obs_components=("screens", "screens"))


def test_empty_component_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        KantoConfig(rom_path="fake.gb", obs_components=())


def test_list_is_coerced_to_tuple() -> None:
    """YAML/JSON configs provide lists; the field must stay a tuple."""
    config = KantoConfig(rom_path="fake.gb", obs_components=["screens", "health"])
    assert config.obs_components == ("screens", "health")


# =============================================================================
# ENVIRONMENT SPACES
# =============================================================================


def test_default_env_space_is_unchanged() -> None:
    """The default tier must reproduce the historical observation exactly."""
    env = KantoRedEnv(config=KantoConfig(rom_path="fake.gb"))
    try:
        # gymnasium's Dict sorts keys, so compare as a set rather than order.
        assert set(env.observation_space.spaces) == set(DEFAULT_OBS_COMPONENTS)
        assert env.observation_space["screens"].shape == (3, 72, 80)
        assert env.observation_space["events"].n == EVENT_OBS_DIM
        assert env.observation_space["map"].shape == MAP_OBS_SHAPE
    finally:
        env.close()


def test_pixels_only_env_space() -> None:
    config = KantoConfig(
        rom_path="fake.gb",
        obs_components=("screens",),
        frame_stacks=2,
        screen_size=(36, 40),
    )
    env = KantoRedEnv(config=config)
    try:
        assert list(env.observation_space.spaces) == ["screens"]
        assert env.observation_space["screens"].shape == (2, 36, 40)
        assert env._want_screens is True
    finally:
        env.close()


def test_ram_only_tier_skips_screen_buffer() -> None:
    config = KantoConfig(
        rom_path="fake.gb",
        obs_components=("health", "events"),
    )
    env = KantoRedEnv(config=config)
    try:
        assert set(env.observation_space.spaces) == {"health", "events"}
        assert env._want_screens is False
    finally:
        env.close()


# =============================================================================
# PRODUCER SPACES AND READS
# =============================================================================


def test_screen_producer_returns_frame_stack() -> None:
    producer = ScreenProducer(frame_stacks=3, screen_size=(72, 80))
    assert producer.space().shape == (3, 72, 80)
    ctx = make_context()
    assert producer.read(ctx) is ctx.frame_stack


def test_health_producer_normalizes_and_handles_empty_party() -> None:
    producer = HealthProducer()
    half = producer.read(make_context(current_hp=50, max_hp=100))
    assert half.dtype == np.float32
    assert half.shape == (1,)
    assert half[0] == pytest.approx(0.5)

    empty = producer.read(make_context(current_hp=0, max_hp=0))
    assert empty[0] == pytest.approx(0.0)


def test_level_producer_fourier_encoding_and_cache() -> None:
    producer = LevelProducer(dim=8)
    levels = (50,)
    first = producer.read(make_context(levels=levels))

    avg = np.mean(levels) / 100.0
    freqs = np.arange(1, 5)
    expected = np.concatenate(
        [np.sin(2 * np.pi * freqs * avg), np.cos(2 * np.pi * freqs * avg)]
    ).astype(np.float32)
    np.testing.assert_allclose(first, expected)

    # Same levels must hit the cache and return the same array object.
    second = producer.read(make_context(levels=levels))
    assert second is first

    # reset() must clear the cache.
    producer.reset()
    third = producer.read(make_context(levels=levels))
    assert third is not first
    np.testing.assert_allclose(third, first)


def test_level_producer_empty_party_is_zero() -> None:
    producer = LevelProducer()
    assert np.array_equal(producer.read(make_context(levels=())), np.zeros(8))


def test_flag_and_map_spaces() -> None:
    assert EventsProducer().space().n == EVENT_OBS_DIM
    assert MapProducer().space().shape == MAP_OBS_SHAPE
    assert LocalMapProducer().space().shape == (LOCAL_MAP_SIZE, LOCAL_MAP_SIZE)
    assert list(RecentActionsProducer().space().nvec) == [8] * ACTION_HISTORY_LEN


def test_local_map_is_centered_and_map_scoped() -> None:
    center = LOCAL_MAP_SIZE // 2
    visits = {(61, 10, 10), (61, 11, 10), (61, 10, 11), (62, 10, 10)}

    view = build_local_map(visits, 61, 10, 10)
    assert view.dtype == np.uint8
    assert view[center, center] == 128
    assert view[center, center + 1] == 255  # (11, 10), right of the player
    assert view[center + 1, center] == 255  # (10, 11), below the player
    assert view[center, center - 1] == 0
    assert np.sum(view == 255) == 2  # map 62's visit must not leak in

    moved = build_local_map(visits, 61, 11, 10)
    assert moved[center, center] == 128
    assert moved[center, center - 1] == 255  # (10, 10) is now left of player
    assert moved[center, center + 1] == 0


def test_local_map_producer_reads_context() -> None:
    window = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE), dtype=np.uint8)
    producer = LocalMapProducer()
    assert producer.read(make_context(local_map=window)) is window


def test_env_space_includes_opt_in_local_map() -> None:
    config = KantoConfig(
        rom_path="fake.gb",
        obs_components=(*DEFAULT_OBS_COMPONENTS, "local_map"),
    )
    env = KantoRedEnv(config=config)
    try:
        assert env.observation_space["local_map"].shape == (
            LOCAL_MAP_SIZE,
            LOCAL_MAP_SIZE,
        )
    finally:
        env.close()


def test_producers_raise_when_context_field_is_missing() -> None:
    with pytest.raises(ValueError, match="frame_stack"):
        ScreenProducer(3, (72, 80)).read(make_context(frame_stack=None))
    with pytest.raises(ValueError, match="event_flags"):
        EventsProducer().read(make_context(event_flags=None))
    with pytest.raises(ValueError, match="badge_flags"):
        from kantorl.obs.producers import BadgesProducer

        BadgesProducer().read(make_context(badge_flags=None))


# =============================================================================
# END-TO-END OBSERVATION WIRING (MOCKED EMULATOR)
# =============================================================================


def test_default_env_observation_matches_declared_space(mock_env) -> None:
    """Every emitted array must be inside its declared space."""
    obs, _ = mock_env.reset()
    assert set(obs) == set(mock_env.observation_space.spaces)
    for key, space in mock_env.observation_space.spaces.items():
        assert space.contains(obs[key]), f"{key!r} is outside {space}"


def test_pixels_only_env_observation(mock_pyboy) -> None:
    from unittest.mock import patch

    config = KantoConfig(rom_path="fake.gb", obs_components=("screens",))
    with patch("kantorl.env.PyBoy", return_value=mock_pyboy):
        env = KantoRedEnv(config=config)
        try:
            obs, _ = env.reset()
            assert set(obs) == {"screens"}
            assert obs["screens"].shape == (3, 72, 80)
        finally:
            env.close()
