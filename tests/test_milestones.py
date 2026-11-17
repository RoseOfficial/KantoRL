"""Tests for checkpoint loading and milestone tracking.

No ROM or torch is needed: milestone tracking is pure bookkeeping over the
environment's info dict, and checkpoint metadata is read from a synthetic zip.
"""

from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from kantorl.checkpoints import (
    env_kwargs_from_model,
    is_recurrent_checkpoint,
    observation_components,
    read_checkpoint_metadata,
)
from kantorl.milestones import (
    BADGE_LABELS,
    CORE_MILESTONE_KEYS,
    EpisodeResult,
    MilestoneReport,
    MilestoneTracker,
    badge_milestone,
    event_milestone,
    map_milestone,
    warn_if_deterministic_seeds,
    write_episodes_csv,
    write_report_json,
    write_summary_csv,
)

# =============================================================================
# CHECKPOINT METADATA
# =============================================================================


def make_checkpoint(path: Path, policy_class: str) -> Path:
    """Write a minimal sb3-shaped zip containing only the metadata entry."""
    metadata = {"policy_class": policy_class, "policy_kwargs": {}}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(metadata))
    return path


def test_reads_metadata_and_detects_ppo(tmp_path: Path) -> None:
    policy_class = "stable_baselines3.common.policies.ActorCriticPolicy"
    path = make_checkpoint(tmp_path / "ppo.zip", policy_class)
    assert read_checkpoint_metadata(path)["policy_class"].startswith("stable_baselines3")
    assert is_recurrent_checkpoint(path) is False


def test_detects_recurrent_policy(tmp_path: Path) -> None:
    path = make_checkpoint(
        tmp_path / "rec.zip",
        "sb3_contrib.common.recurrent.policies.RecurrentMultiInputActorCriticPolicy",
    )
    assert is_recurrent_checkpoint(path) is True


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_checkpoint_metadata(tmp_path / "nope.zip")


def test_non_sb3_zip_raises(tmp_path: Path) -> None:
    path = tmp_path / "other.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("something.txt", "hello")
    with pytest.raises(ValueError, match="not a Stable-Baselines3 checkpoint"):
        read_checkpoint_metadata(path)


class FakeModel:
    """Minimal stand-in exposing only the ``observation_space`` attribute."""

    def __init__(self, space):
        self.observation_space = space


def test_observation_components_from_space() -> None:
    from gymnasium import spaces

    space = spaces.Dict(
        {
            "screens": spaces.Box(0, 255, (4, 36, 40), dtype=np.uint8),
            "health": spaces.Box(0, 1, (1,), dtype=np.float32),
        }
    )
    model = FakeModel(space)

    assert set(observation_components(model)) == {"screens", "health"}

    kwargs = env_kwargs_from_model(model)
    assert kwargs["frame_stacks"] == 4
    assert kwargs["screen_size"] == (36, 40)


def test_observation_components_ignores_unknown_keys() -> None:
    from gymnasium import spaces

    space = spaces.Dict(
        {
            "screens": spaces.Box(0, 255, (3, 72, 80), dtype=np.uint8),
            "agent_quest": spaces.Box(-1, 1, (5,), dtype=np.float32),
        }
    )
    assert observation_components(FakeModel(space)) == ("screens",)


# =============================================================================
# MILESTONE TRACKING
# =============================================================================


def test_badge_labels_cover_eight_badges() -> None:
    assert len(BADGE_LABELS) == 8
    assert badge_milestone(1).label == "Boulder Badge"
    assert badge_milestone(8).key == "badge:8"


def test_event_and_map_labels_are_human_readable() -> None:
    # Indices from pret/pokered: 37 is EVENT_GOT_POKEDEX, 34 is GOT_STARTER.
    assert event_milestone(37).label == "Got Pokédex"
    assert event_milestone(34).label == "Got Starter"
    assert map_milestone(0).label == "Pallet Town"


def test_tracker_records_first_step_only() -> None:
    tracker = MilestoneTracker()

    tracker.update({"step": 0, "map_id": 0, "badges": 0, "events": 0, "new_event_indices": []})
    assert tracker.first_steps == {"map:0": 0}

    # Same map again: no duplicate record.
    tracker.update({"step": 10, "map_id": 0, "badges": 0, "events": 0, "new_event_indices": []})
    assert tracker.first_steps == {"map:0": 0}

    # The new game's initial block of flags is a baseline, not an achievement.
    tracker.update(
        {"step": 12, "map_id": 0, "badges": 0, "events": 8, "new_event_indices": INIT_EVENTS}
    )
    assert "map:0" in tracker.first_steps
    assert not [key for key in tracker.first_steps if key.startswith("event:")]

    tracker.update({"step": 20, "map_id": 0, "badges": 1, "events": 9, "new_event_indices": [3]})
    assert tracker.first_steps["badge:1"] == 20
    assert tracker.first_steps["event:3"] == 20

    # Re-entering the badge/event must not overwrite the timestamp.
    tracker.update({"step": 99, "map_id": 0, "badges": 1, "events": 9, "new_event_indices": [3]})
    assert tracker.first_steps["badge:1"] == 20
    assert tracker.first_steps["event:3"] == 20


def test_tracker_emits_hits_only_once() -> None:
    tracker = MilestoneTracker()
    tracker.update(
        {"step": 4, "map_id": 12, "badges": 0, "events": 8, "new_event_indices": INIT_EVENTS}
    )

    first = tracker.update(
        {"step": 5, "map_id": 12, "badges": 0, "events": 10, "new_event_indices": [1, 2]}
    )
    assert {hit.milestone.key for hit in first} == {"event:1", "event:2"}

    second = tracker.update(
        {"step": 6, "map_id": 12, "badges": 0, "events": 10, "new_event_indices": [1]}
    )
    assert second == []


def test_tracker_backfills_skipped_badges() -> None:
    """A checkpoint that starts with badges already set still reports them."""
    tracker = MilestoneTracker()
    tracker.update({"step": 0, "map_id": 0, "badges": 3, "new_event_indices": []})
    assert {key for key in tracker.first_steps if key.startswith("badge:")} == {
        "badge:1",
        "badge:2",
        "badge:3",
    }


def test_tracker_reset_clears_everything() -> None:
    tracker = MilestoneTracker()
    tracker.update({"step": 1, "map_id": 0, "badges": 0, "new_event_indices": [0]})
    tracker.reset()
    assert tracker.first_steps == {}
    assert tracker.catalog == {}


# =============================================================================
# REPORT AND SERIALIZATION
# =============================================================================


def make_report() -> MilestoneReport:
    catalog = {key: _milestone(key) for key in CORE_MILESTONE_KEYS}
    catalog["event:3"] = event_milestone(3)
    catalog["map:0"] = map_milestone(0)

    episodes = [
        EpisodeResult(
            episode=1,
            seed=0,
            steps=100,
            reward=1.5,
            badges=1,
            events=4,
            explore_tiles=30,
            party_count=1,
            first_steps={"event:0": 0, "event:3": 40, "badge:1": 90, "map:0": 0},
        ),
        EpisodeResult(
            episode=2,
            seed=1,
            steps=100,
            reward=0.5,
            badges=0,
            events=2,
            explore_tiles=10,
            party_count=1,
            first_steps={"event:0": 0, "map:0": 0},
        ),
    ]
    return MilestoneReport(
        checkpoint="runs/model_100.zip",
        obs_components=("screens", "health"),
        budget=100,
        deterministic=True,
        episodes=episodes,
        catalog=catalog,
    )


def _milestone(key: str):
    kind, _, index = key.partition(":")
    return {
        "badge": badge_milestone,
        "event": event_milestone,
        "map": map_milestone,
    }[kind](int(index))


def test_summary_distinguishes_never_reached_from_step_zero() -> None:
    report = make_report()
    rows = {row["milestone_key"]: row for row in report.summary_rows()}

    # Reached at step 0 in both episodes.
    assert rows["event:0"]["median_first_step"] == 0
    assert rows["event:0"]["reach_rate"] == 1.0

    # Reached once out of two runs.
    assert rows["badge:1"]["reached_count"] == 1
    assert rows["badge:1"]["median_first_step"] == 90.0

    # Never reached: present with an explicit None, not a zero.
    assert rows["badge:8"]["reached_count"] == 0
    assert rows["badge:8"]["median_first_step"] is None
    assert rows["badge:8"]["reach_rate"] == 0.0


def test_core_milestones_are_always_present() -> None:
    report = make_report()
    keys = {row["milestone_key"] for row in report.summary_rows()}
    assert set(CORE_MILESTONE_KEYS) <= keys


def test_json_round_trips(tmp_path: Path) -> None:
    report = make_report()
    path = write_report_json(report, tmp_path / "milestones.json")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["obs_components"] == ["screens", "health"]
    assert data["budget"] == 100
    assert len(data["episodes"]) == 2
    assert data["episodes"][0]["first_steps"]["event:3"] == 40
    assert data["catalog"]["badge:1"]["label"] == "Boulder Badge"


def test_episode_csv_has_one_row_per_reached_milestone(tmp_path: Path) -> None:
    report = make_report()
    path = write_episodes_csv(report, tmp_path / "episodes.csv")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split(",")
    assert "milestone_key" in header
    assert "first_step" in header
    # 4 milestones in episode 1 + 2 in episode 2.
    assert len(lines) == 1 + 6


def test_summary_csv_one_row_per_milestone(tmp_path: Path) -> None:
    report = make_report()
    path = write_summary_csv(report, tmp_path / "summary.csv")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 + len(report.summary_rows())
    assert "reach_rate" in lines[0]


def test_format_summary_is_readable() -> None:
    text = make_report().format_summary()
    assert "median step" in text
    assert "Boulder Badge" in text


def test_evaluate_rejects_empty_seeds_before_loading() -> None:
    from kantorl.milestones import evaluate_milestones

    # Validation runs before the checkpoint is touched, so a bogus path is fine.
    with pytest.raises(ValueError, match="at least one episode seed"):
        evaluate_milestones("missing.zip", "missing.gb", seeds=())


def test_evaluate_rejects_nonpositive_budget() -> None:
    from kantorl.milestones import evaluate_milestones

    with pytest.raises(ValueError, match="budget must be positive"):
        evaluate_milestones("missing.zip", "missing.gb", seeds=(0,), budget=0)


# =============================================================================
# ROUND TRIP THROUGH A REAL SB3 CHECKPOINT
# =============================================================================


class DummyEnv(gym.Env):
    """Gymnasium-shaped env that never touches an emulator.

    Used only to give Stable-Baselines3 a real Dict observation space to record
    in a checkpoint. It reuses the actual environment's spaces so the saved
    metadata matches what training would write.
    """

    def __init__(self, observation_space, action_space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        return _zero_obs(self.observation_space), {}

    def step(self, action):
        return _zero_obs(self.observation_space), 0.0, False, False, {}

    def render(self):
        return None

    def close(self):
        return None


def _zero_obs(space) -> dict:
    return {key: np.zeros(sub.shape, dtype=sub.dtype) for key, sub in space.spaces.items()}


def test_sb3_checkpoint_round_trip_preserves_tier_and_shapes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from stable_baselines3 import PPO

    from kantorl.config import KantoConfig
    from kantorl.env import KantoRedEnv

    # A pixels-only, 4-frame, downscaled model: every inferred field differs
    # from the default config, so a silent fallback would be visible.
    config = KantoConfig(
        rom_path="fake.gb",
        obs_components=("screens",),
        frame_stacks=4,
        screen_size=(36, 40),
    )
    env = KantoRedEnv(config=config)
    try:
        dummy = DummyEnv(env.observation_space, env.action_space)
        model = PPO(
            "MultiInputPolicy",
            dummy,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            device="cpu",
        )
    finally:
        env.close()

    path = tmp_path / "pixels.zip"
    model.save(path)

    from kantorl.checkpoints import env_kwargs_from_model, load_model

    assert is_recurrent_checkpoint(path) is False
    loaded = load_model(path, device="cpu")

    assert observation_components(loaded) == ("screens",)
    assert env_kwargs_from_model(loaded) == {
        "obs_components": ("screens",),
        "frame_stacks": 4,
        "screen_size": (36, 40),
    }


# =============================================================================
# DETERMINISTIC-SEED WARNING
# =============================================================================


def test_warns_when_deterministic_eval_gets_several_seeds():
    """Argmax on a deterministic emulator makes extra seeds copies, not samples."""
    with pytest.warns(UserWarning, match="replay the same episode"):
        warn_if_deterministic_seeds(True, (0, 1))


def test_single_seed_is_not_noisy():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_deterministic_seeds(True, (0,))


def test_stochastic_eval_is_not_noisy():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_deterministic_seeds(False, (0, 1, 2))


# =============================================================================
# GAME-INITIALIZATION EVENT FLAGS
# =============================================================================

# Observed together on a fresh new game, before the player can act.
INIT_EVENTS = [0, 25, 32, 33, 34, 35, 39, 960]


def test_initialization_batch_is_not_counted_as_achievements():
    """A new game sets a block of flags at once; those are not progress."""
    tracker = MilestoneTracker()

    tracker.update(
        {
            "step": 16,
            "map_id": 0,
            "badges": 0,
            "events": 8,
            "new_event_indices": INIT_EVENTS,
        }
    )

    assert not [k for k in tracker.first_steps if k.startswith("event:")]
    # The map in the same step is still recorded; only events are baselined.
    assert tracker.first_steps == {"map:0": 16}


def test_real_event_after_initialization_is_recorded():
    tracker = MilestoneTracker()

    tracker.update(
        {"step": 16, "map_id": 0, "badges": 0, "events": 8, "new_event_indices": INIT_EVENTS}
    )
    tracker.update({"step": 900, "map_id": 12, "badges": 0, "events": 9, "new_event_indices": [50]})

    assert tracker.first_steps["event:50"] == 900


def test_baseline_is_rearmed_by_reset():
    """Each episode starts a new game, so the next batch is a baseline again."""
    tracker = MilestoneTracker()
    tracker.update(
        {"step": 16, "map_id": 0, "badges": 0, "events": 8, "new_event_indices": INIT_EVENTS}
    )

    tracker.reset()
    tracker.update({"step": 16, "map_id": 0, "badges": 0, "events": 8, "new_event_indices": [50]})

    assert not [k for k in tracker.first_steps if k.startswith("event:")]
