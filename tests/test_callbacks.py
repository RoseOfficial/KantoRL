"""Tests for the thread-tuning callback.

The contract: the rollout phase gets ``rollout_threads`` and the update phase
gets ``update_threads``, because the two phases run sequentially with opposite
CPU appetites. A regression here is invisible in training results -- it only
silently costs throughput -- so it is worth pinning down.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kantorl.callbacks import (
    CumulativeCheckpointCallback,
    FirstSeenCallback,
    TensorboardCallback,
    ThreadTuningCallback,
)


@pytest.fixture(autouse=True)
def restore_thread_count():
    """Leave the global torch thread count as we found it."""
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


def make_callback(device: str = "cpu", **kwargs) -> ThreadTuningCallback:
    """Build a callback with a stand-in model exposing only what it reads."""
    callback = ThreadTuningCallback(**kwargs)
    callback.model = SimpleNamespace(device=device)
    return callback


def test_rollout_and_update_use_different_thread_counts():
    callback = make_callback(rollout_threads=2, update_threads=8)

    callback._on_rollout_start()
    assert torch.get_num_threads() == 2

    callback._on_rollout_end()
    assert torch.get_num_threads() == 8


def test_training_start_applies_rollout_threads():
    callback = make_callback(rollout_threads=3, update_threads=8)

    callback._on_training_start()

    assert torch.get_num_threads() == 3


def test_update_threads_default_to_every_logical_core():
    callback = make_callback(rollout_threads=2)

    callback._on_rollout_end()

    assert torch.get_num_threads() == os.cpu_count()


def test_non_cpu_device_is_left_alone():
    original = torch.get_num_threads()
    callback = make_callback(device="cuda:0", rollout_threads=1, update_threads=16)

    callback._on_rollout_start()
    callback._on_rollout_end()

    assert torch.get_num_threads() == original


def test_on_step_continues_training():
    assert make_callback()._on_step() is True


# =============================================================================
# CHECKPOINT RETENTION
# =============================================================================


def test_cumulative_checkpoints_keep_recent_window_and_lineage_marks(tmp_path):
    callback = CumulativeCheckpointCallback(
        tmp_path,
        save_freq=100_000,
        keep_latest=3,
        keep_every=1_000_000,
        verbose=0,
    )
    steps = (900_000, 1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000, 1_500_000)
    for step in steps:
        (tmp_path / f"model_{step}.zip").write_bytes(b"checkpoint")
    (tmp_path / "best_model.zip").write_bytes(b"keep")
    (tmp_path / "badge_1_model.zip").write_bytes(b"keep")

    callback._prune()

    remaining = sorted(path.name for path in tmp_path.glob("model_*.zip"))
    assert remaining == [
        "model_1000000.zip",  # 1M lineage mark
        "model_1300000.zip",  # recent window
        "model_1400000.zip",
        "model_1500000.zip",
    ]
    assert (tmp_path / "best_model.zip").exists()
    assert (tmp_path / "badge_1_model.zip").exists()


def test_cumulative_prune_ignores_unparseable_names(tmp_path):
    callback = CumulativeCheckpointCallback(
        tmp_path, keep_latest=1, keep_every=1_000_000, verbose=0
    )
    (tmp_path / "model_1000000.zip").write_bytes(b"keep")
    (tmp_path / "model_latest.zip").write_bytes(b"keep")

    callback._prune()

    assert (tmp_path / "model_latest.zip").exists()


# =============================================================================
# FIRST-SEEN MILESTONES
# =============================================================================


def test_first_seen_writes_each_milestone_once(tmp_path):
    callback = FirstSeenCallback(session_path=tmp_path)
    callback._on_training_start()

    callback.num_timesteps = 1_234
    callback.locals = {
        "infos": [{"badges": 1, "new_event_indices": [34, 35]}]
    }
    assert callback._on_step() is True

    # Same event again, plus the second badge: only the badge is new.
    callback.num_timesteps = 5_678
    callback.locals = {
        "infos": [{"badges": 2, "new_event_indices": [34]}]
    }
    assert callback._on_step() is True

    rows = (tmp_path / "first_seen.csv").read_text().strip().splitlines()

    assert rows[0] == "step,kind,index,label"
    assert len(rows) == 5  # header + 2 events + 2 badges
    assert rows[1] == "1234,badge,1,Boulder Badge"
    assert rows[2].startswith("1234,event,34,")
    assert rows[3].startswith("1234,event,35,")
    assert rows[4] == "5678,badge,2,Cascade Badge"


def test_first_seen_survives_a_restart(tmp_path):
    """Resuming appends; it must not truncate the record of earlier progress."""
    first = FirstSeenCallback(session_path=tmp_path)
    first._on_training_start()
    first.num_timesteps = 10
    first.locals = {"infos": [{"badges": 0, "new_event_indices": [34]}]}
    first._on_step()

    resumed = FirstSeenCallback(session_path=tmp_path)
    resumed._on_training_start()
    resumed.num_timesteps = 20
    resumed.locals = {
        "infos": [{"badges": 1, "new_event_indices": [34]}]
    }
    resumed._on_step()

    rows = (tmp_path / "first_seen.csv").read_text().strip().splitlines()
    assert len(rows) == 3  # header + the event + the badge, no duplicates


def test_tensorboard_badge_max_is_all_time_not_windowed(tmp_path):
    """Seed 44 earned a badge and the windowed metric still reported zero."""
    callback = TensorboardCallback(log_freq=1000, session_path=tmp_path)

    callback.num_timesteps = 1
    callback.locals = {"infos": [{"badges": 1}]}
    callback._on_step()
    assert callback._max_badges_seen == 1

    # Simulate the post-log clear, then a badge-free window.
    callback.badges_collected.clear()
    callback.num_timesteps = 2
    callback.locals = {"infos": [{"badges": 0}]}
    callback._on_step()

    assert callback._max_badges_seen == 1
    assert max(callback.badges_collected) == 0


# =============================================================================
# BEST-CHECKPOINT RETENTION
# =============================================================================


class _FakeEnv:
    def __init__(self, calls):
        self.calls = calls

    def save(self, path):
        self.calls.append(("env", path))


class _FakeModel:
    def __init__(self):
        self.calls = []
        self.env = _FakeEnv(self.calls)

    def save(self, path):
        self.calls.append(("model", path))

    def get_env(self):
        return self.env


def test_first_seen_saves_best_and_badge_snapshots(tmp_path):
    callback = FirstSeenCallback(session_path=tmp_path)
    callback.model = _FakeModel()
    callback._on_training_start()

    callback.num_timesteps = 1_234
    callback.locals = {
        "infos": [{"badges": 1, "new_event_indices": [34]}]
    }
    assert callback._on_step() is True

    saved = [Path(path).name for kind, path in callback.model.calls if kind == "model"]
    assert "badge_1_model.zip" in saved
    assert "best_model.zip" in saved
    assert (tmp_path / "checkpoints" / "badge_1_model_step.txt").exists()
    assert (tmp_path / "checkpoints" / "best_model_step.txt").read_text().startswith(
        "1234,"
    )


def test_first_seen_does_not_snapshot_on_duplicates(tmp_path):
    callback = FirstSeenCallback(session_path=tmp_path)
    callback.model = _FakeModel()
    callback._on_training_start()

    callback.num_timesteps = 10
    callback.locals = {"infos": [{"badges": 0, "new_event_indices": [34]}]}
    callback._on_step()
    saves = len(callback.model.calls)

    callback.num_timesteps = 20
    callback.locals = {"infos": [{"badges": 0, "new_event_indices": [34]}]}
    callback._on_step()

    assert len(callback.model.calls) == saves


def test_first_seen_snapshots_can_be_disabled(tmp_path):
    callback = FirstSeenCallback(session_path=tmp_path, save_best=False)
    callback.model = _FakeModel()
    callback._on_training_start()

    callback.num_timesteps = 10
    callback.locals = {"infos": [{"badges": 1, "new_event_indices": [34]}]}
    callback._on_step()

    assert callback.model.calls == []
    assert not (tmp_path / "checkpoints").exists()
