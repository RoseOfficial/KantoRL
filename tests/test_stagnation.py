"""Tests for the stagnation tracker.

The contract: an episode ends once it has gone ``window`` steps without
collecting ``threshold`` reward. This is what stops a stalled agent from
spending the whole horizon, which is how the earlier flat-reward runs wasted
~10k steps at a time.
"""

from __future__ import annotations

from kantorl.env import StagnationTracker


def test_truncates_after_a_reward_free_window():
    tracker = StagnationTracker(window=5, threshold=1.0)

    assert [tracker.update(0.0) for _ in range(4)] == [False, False, False, False]
    assert tracker.update(0.0) is True


def test_progress_resets_the_window():
    tracker = StagnationTracker(window=5, threshold=1.0)

    for _ in range(4):
        tracker.update(0.0)

    # Crossing the threshold counts as progress and restarts the count.
    assert tracker.update(1.0) is False
    assert [tracker.update(0.0) for _ in range(4)] == [False, False, False, False]
    assert tracker.update(0.0) is True


def test_small_rewards_accumulate_toward_the_threshold():
    """Breadcrumbs count: 0.3/step reaches 1.0 on the fourth step."""
    tracker = StagnationTracker(window=5, threshold=1.0)

    assert [tracker.update(0.3) for _ in range(3)] == [False, False, False]
    assert tracker.update(0.3) is False  # 1.2 >= 1.0, window restarts
    assert tracker.steps == 0


def test_zero_window_disables_the_check():
    tracker = StagnationTracker(window=0, threshold=1.0)

    assert all(tracker.update(0.0) is False for _ in range(100))


def test_reset_clears_the_window():
    tracker = StagnationTracker(window=5, threshold=1.0)

    for _ in range(4):
        tracker.update(0.0)
    tracker.reset()

    assert [tracker.update(0.0) for _ in range(4)] == [False, False, False, False]
