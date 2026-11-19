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


# =============================================================================
# LEVEL REWARD
# =============================================================================


def test_level_potential_is_linear_then_capped():
    assert DefaultReward.level_potential(0) == 0.0
    assert DefaultReward.level_potential(10) == 10.0
    assert DefaultReward.level_potential(15) == 30.0
    # Above 15, growth slows to a quarter rate.
    assert DefaultReward.level_potential(19) == pytest.approx(31.0)


def test_first_step_does_not_pay_for_the_starting_levels():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(GameState(map_id=0, x=0, y=0, level_sum=5), None, action=0)

    assert reward.last_rewards["level"] == 0.0


def test_levelling_up_pays_the_potential_difference():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(GameState(map_id=0, x=0, y=0, level_sum=5), None, action=0)
    reward.calculate(GameState(map_id=0, x=0, y=0, level_sum=6), None, action=0)

    assert reward.last_rewards["level"] == pytest.approx(1.0)


def test_level_gains_above_the_cap_are_worth_a_quarter():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(GameState(map_id=0, x=0, y=0, level_sum=15), None, action=0)
    reward.calculate(GameState(map_id=0, x=0, y=0, level_sum=19), None, action=0)

    assert reward.last_rewards["level"] == pytest.approx(1.0)


def test_losing_levels_does_not_allow_re_earning_them():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(GameState(map_id=0, x=0, y=0, level_sum=10), None, action=0)
    reward.calculate(GameState(map_id=0, x=0, y=0, level_sum=8), None, action=0)
    reward.calculate(GameState(map_id=0, x=0, y=0, level_sum=10), None, action=0)

    assert reward.last_rewards["level"] == 0.0


def test_level_tracking_resets_between_episodes():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)

    reward.calculate(GameState(map_id=0, x=0, y=0, level_sum=20), None, action=0)
    reward.reset()
    reward.calculate(GameState(map_id=0, x=0, y=0, level_sum=20), None, action=0)

    assert reward.last_rewards["level"] == 0.0


def test_create_reward_forwards_level_weight():
    assert create_reward("default", level_weight=0.25).level_weight == 0.25


def test_new_event_refreshes_coordinate_novelty():
    """Progress reopens ground: otherwise the coordinate reward stays spent."""
    reward = DefaultReward(reward_scale=1.0, explore_weight=1.0)
    here = GameState(map_id=0, x=1, y=1, event_count=7)

    # Baseline the event counter at the game-initialization block.
    reward.calculate(here, None, action=0)
    assert reward.last_rewards["explore"] == pytest.approx(0.1)

    # Same tile again: already visited, so nothing.
    reward.calculate(here, None, action=0)
    assert reward.last_rewards["explore"] == 0.0

    # Gaining an event clears the visited set, so the same tile pays again.
    gained = GameState(map_id=0, x=1, y=1, event_count=8)
    reward.calculate(gained, None, action=0)

    assert reward.last_rewards["event"] == pytest.approx(8.0)
    assert reward.last_rewards["explore"] == pytest.approx(0.1)


def test_healing_is_off_by_default():
    """Losing to black out and heal is farmable, so it pays nothing by default."""
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0)
    hurt = GameState(map_id=0, x=0, y=0, current_hp=5, max_hp=10)
    healed = GameState(map_id=0, x=0, y=0, current_hp=10, max_hp=10)

    reward.calculate(hurt, None, action=0)
    reward.calculate(healed, hurt, action=0)

    assert reward.last_rewards["heal"] == 0.0


def test_healing_pays_when_explicitly_enabled():
    reward = DefaultReward(reward_scale=1.0, explore_weight=0.0, heal_weight=1.0)
    hurt = GameState(map_id=0, x=0, y=0, current_hp=5, max_hp=10)
    healed = GameState(map_id=0, x=0, y=0, current_hp=10, max_hp=10)

    reward.calculate(hurt, None, action=0)
    reward.calculate(healed, hurt, action=0)

    assert reward.last_rewards["heal"] == pytest.approx(5.0)


def test_healing_pays_once_per_map_when_bounded():
    """Stops blackout farming while still pulling toward new centres."""
    reward = DefaultReward(
        reward_scale=1.0, explore_weight=0.0, heal_weight=1.0, heal_once_per_map=True
    )
    hurt = GameState(map_id=5, x=0, y=0, current_hp=5, max_hp=10)
    healed = GameState(map_id=5, x=0, y=0, current_hp=10, max_hp=10)

    reward.calculate(hurt, None, action=0)
    reward.calculate(healed, hurt, action=0)
    assert reward.last_rewards["heal"] == pytest.approx(5.0)

    # Same map again: a second heal must not pay.
    reward.calculate(hurt, healed, action=0)
    reward.calculate(healed, hurt, action=0)
    assert reward.last_rewards["heal"] == 0.0

    # A new map pays again.
    elsewhere = GameState(map_id=12, x=0, y=0, current_hp=10, max_hp=10)
    reward.calculate(elsewhere, hurt, action=0)
    assert reward.last_rewards["heal"] == pytest.approx(5.0)


def test_unbounded_healing_still_pays_repeatedly_when_opted_in():
    reward = DefaultReward(
        reward_scale=1.0, explore_weight=0.0, heal_weight=1.0, heal_once_per_map=False
    )
    hurt = GameState(map_id=5, x=0, y=0, current_hp=5, max_hp=10)
    healed = GameState(map_id=5, x=0, y=0, current_hp=10, max_hp=10)

    reward.calculate(hurt, None, action=0)
    reward.calculate(healed, hurt, action=0)
    reward.calculate(hurt, healed, action=0)
    reward.calculate(healed, hurt, action=0)

    assert reward.last_rewards["heal"] == pytest.approx(5.0)
