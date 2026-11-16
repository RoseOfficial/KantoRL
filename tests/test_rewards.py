"""Tests for the interaction reward and the stuck-penalty default.

Regression these protect: an agent paid only for movement learns never to press
A, because A freezes the character while dialogue is open. That is fatal when
the starter sits behind several A presses, so interaction has to pay something.
"""

from __future__ import annotations

import pytest

from kantorl.rewards import (
    ACTION_A,
    DefaultReward,
    ExplorationReward,
    GameState,
    create_reward,
)


def state(x: int = 5, y: int = 5, map_id: int = 0) -> GameState:
    return GameState(map_id=map_id, x=x, y=y)


def test_action_a_matches_the_environment_action_index():
    """ACTION_A duplicates a constant in env.py; keep them honest."""
    from pyboy.utils import WindowEvent

    from kantorl.env import PRESS_ACTIONS

    assert PRESS_ACTIONS[ACTION_A] == WindowEvent.PRESS_BUTTON_A


def test_pressing_a_somewhere_new_pays_once():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)
    here = state(x=7, y=9)

    reward.calculate(here, None, action=ACTION_A)
    assert reward.last_rewards["interact"] == pytest.approx(0.1)

    reward.calculate(here, None, action=ACTION_A)
    assert reward.last_rewards["interact"] == 0.0


def test_interaction_pays_again_at_a_new_position():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(state(x=7, y=9), None, action=ACTION_A)
    reward.calculate(state(x=8, y=9), None, action=ACTION_A)

    assert reward.last_rewards["interact"] == pytest.approx(0.1)


def test_moving_does_not_earn_the_interaction_reward():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(state(x=7, y=9), None, action=1)  # UP

    assert reward.last_rewards["interact"] == 0.0


def test_reward_scales_with_interact_weight():
    reward = DefaultReward(reward_scale=2.0, explore_weight=0.0, interact_weight=0.5)

    reward.calculate(state(), None, action=ACTION_A)

    assert reward.last_rewards["interact"] == pytest.approx(2.0 * 0.5 * 0.1)


def test_interaction_tracking_is_cleared_between_episodes():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)
    here = state()

    reward.calculate(here, None, action=ACTION_A)
    reward.reset()
    reward.calculate(here, None, action=ACTION_A)

    assert reward.last_rewards["interact"] == pytest.approx(0.1)


def test_exploration_reward_also_pays_for_interaction():
    reward = ExplorationReward(reward_scale=1.0)

    reward.calculate(state(), None, action=ACTION_A)

    assert reward.last_rewards["interact"] == pytest.approx(0.1)


def test_floor_on_a_reward_free_run_does_not_penalise_by_default():
    """Dialogue and menus are reward-free; punishing them is why A was avoided."""
    reward = DefaultReward()
    here = state()

    for _ in range(600):
        reward.calculate(here, None, action=0)

    assert reward.last_rewards["stuck"] == 0.0


def test_stuck_penalty_still_works_when_opted_into():
    reward = DefaultReward(reward_scale=1.0, stuck_penalty=0.05)
    here = state()

    for _ in range(600):
        reward.calculate(here, None, action=0)

    assert reward.last_rewards["stuck"] == pytest.approx(-0.05)


def test_create_reward_forwards_weights_for_both_rewards():
    default = create_reward("default", interact_weight=0.25, stuck_penalty=0.1)
    exploration = create_reward("exploration", interact_weight=0.25)

    assert default.interact_weight == 0.25
    assert default.stuck_penalty == 0.1
    assert exploration.interact_weight == 0.25


# =============================================================================
# DELTA BASELINE
# =============================================================================


def test_preexisting_events_do_not_pay():
    """A new game sets ~8 flags at init; they are not progress."""
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(GameState(map_id=0, x=0, y=0, event_count=8), None, action=0)

    assert reward.last_rewards["event"] == 0.0


def test_events_earned_after_the_baseline_pay():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(GameState(map_id=0, x=0, y=0, event_count=8), None, action=0)
    reward.calculate(GameState(map_id=0, x=0, y=0, event_count=9), None, action=0)

    assert reward.last_rewards["event"] == pytest.approx(8.0)


def test_preexisting_badges_do_not_pay():
    """A save state can start with badges already earned."""
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(GameState(map_id=0, x=0, y=0, badges=3), None, action=0)

    assert reward.last_rewards["badge"] == 0.0


def test_reset_rebaselines_for_the_next_episode():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(GameState(map_id=0, x=0, y=0, event_count=8), None, action=0)
    reward.reset()
    reward.calculate(GameState(map_id=0, x=0, y=0, event_count=8), None, action=0)

    assert reward.last_rewards["event"] == 0.0
