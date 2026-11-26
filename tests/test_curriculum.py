"""Tests for the self-generated checkpoint archive.

The archive is the retention mechanism: it must keep both high-progress states
and deep recent states, and its sampler must be able to draw from each. These
tests use dummy state bytes; the emulator is not involved.
"""

from __future__ import annotations

import json
from pathlib import Path

from kantorl.curriculum import CheckpointPool


def add(pool: CheckpointPool, progress: int, step: int) -> None:
    """Add a dummy state with a controlled progress score and save serial."""
    pool.add(
        state_bytes=f"state-{progress}".encode(),
        badges=0,
        event_count=progress,
        map_id=1,
        step=step,
    )


def test_pool_records_the_save_serial(tmp_path: Path) -> None:
    pool = CheckpointPool(tmp_path, max_size=10)
    add(pool, progress=1, step=123)

    meta = next(tmp_path.glob("cp_*.json"))
    assert json.loads(meta.read_text())["step"] == 123


def test_frontier_sampling_draws_from_the_top_half(tmp_path: Path) -> None:
    pool = CheckpointPool(tmp_path, max_size=10)
    for progress in range(1, 7):
        add(pool, progress=progress, step=progress)

    for _ in range(20):
        result = pool.sample(frontier_fraction=1.0, recent_fraction=0.0)
        assert result is not None
        _, metadata = result
        assert metadata["progress"] in {4, 5, 6}


def test_recent_sampling_draws_from_the_latest_third(tmp_path: Path) -> None:
    pool = CheckpointPool(tmp_path, max_size=10)
    for progress in range(1, 7):
        add(pool, progress=progress, step=progress)

    for _ in range(20):
        result = pool.sample(frontier_fraction=0.0, recent_fraction=1.0)
        assert result is not None
        _, metadata = result
        assert metadata["step"] in {5, 6}


def test_pruning_keeps_progress_and_recency(tmp_path: Path) -> None:
    """Pruning on progress alone would discard the deep dungeon state."""
    pool = CheckpointPool(tmp_path, max_size=4)
    for progress in range(1, 21):
        add(pool, progress=progress, step=progress)

    steps = [json.loads(path.read_text())["step"] for path in tmp_path.glob("cp_*.json")]
    progress = [json.loads(path.read_text())["progress"] for path in tmp_path.glob("cp_*.json")]

    # A small hysteresis is allowed so 32 workers do not prune on every add.
    assert len(steps) <= 12
    assert max(steps) == 20
    assert max(progress) == 20
