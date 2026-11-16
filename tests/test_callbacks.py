"""Tests for the thread-tuning callback.

The contract: the rollout phase gets ``rollout_threads`` and the update phase
gets ``update_threads``, because the two phases run sequentially with opposite
CPU appetites. A regression here is invisible in training results -- it only
silently costs throughput -- so it is worth pinning down.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from kantorl.callbacks import ThreadTuningCallback


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
