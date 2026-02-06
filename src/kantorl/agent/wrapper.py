"""
AgentWrapper — main integration point for the modular agent system.

This Gymnasium wrapper combines all agent modules (planner, navigator,
battler, manager, controller) and provides:
1. Mode detection: Determine if we're exploring, battling, or in dialogue
2. Action routing: Override RL actions with scripted behavior when appropriate
3. Observation augmentation: Add battle state, goal, and nav hints to obs
4. Reward shaping: Add potential-based navigation bonuses

Architecture Role:
    AgentWrapper sits in the wrapper chain:
    KantoRedEnv → CurriculumWrapper → AgentWrapper → StreamWrapper

    It intercepts step() to:
    - Detect game mode from memory (battle, dialogue, overworld)
    - Route actions to the appropriate module (battler, controller, RL)
    - Augment observations with agent-specific information
    - Add navigation reward shaping to the environment reward

Modes:
    - EXPLORE: RL policy controls actions. Navigator provides reward shaping.
    - BATTLE: Battler decides, controller executes button sequences.
    - DIALOGUE: Controller presses A to advance text.

Design Decisions:
    - Overrides RL actions during battle/dialogue: The RL policy still
      runs and produces actions, but they're ignored. This creates some
      off-policy data in the rollout buffer, but PPO is robust to this.
    - Observation space extended: Three new keys added to the Dict space.
      This means --agent mode requires --no-resume for first run.
    - Reward shaping is additive: Navigation bonus is added to the
      environment's existing reward, not replacing it.

Dependencies:
    - gymnasium: For the Wrapper base class and spaces
    - numpy: For observation arrays
    - kantorl.memory: For mode detection and battle state reading
    - kantorl.config: For agent configuration fields
    - All agent submodules (planner, navigator, battler, manager, controller)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from kantorl import memory
from kantorl.agent.battler import BattleState, RuleBasedBattler
from kantorl.agent.controller import MenuController
from kantorl.agent.manager import ResourceManager
from kantorl.agent.navigator import Navigator
from kantorl.agent.planner import QuestPlanner
from kantorl.config import KantoConfig

# =============================================================================
# AGENT WRAPPER
# =============================================================================


class AgentWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """
    Gymnasium wrapper integrating the modular agent system.

    Combines quest planning, navigation, battle tactics, resource
    management, and menu control into a single wrapper that augments
    the base environment's observations, actions, and rewards.

    This class is pickle-safe for SubprocVecEnv: all submodules use
    lazy JSON loading with module-level caches, and no unpicklable
    state (file handles, threads, lambdas) is stored.

    Attributes:
        config: KantoConfig with agent settings.
        planner: Quest FSM for strategic goal-setting.
        navigator: Map graph pathfinder and reward shaper.
        battler: Rule-based battle policy.
        manager: Resource management heuristics.
        controller: Scripted button sequence executor.
        mode: Current agent mode ("explore", "battle", "dialogue").
        prev_map: Previous step's map ID (for reward shaping).
        prev_x: Previous step's X coordinate.
        prev_y: Previous step's Y coordinate.
    """

    def __init__(self, env: gym.Env[Any, Any], config: KantoConfig) -> None:
        """
        Initialize the AgentWrapper.

        Args:
            env: Base environment (KantoRedEnv or CurriculumWrapper).
            config: KantoConfig with agent settings.
        """
        super().__init__(env)
        self.config = config

        # Initialize all submodules
        self.planner = QuestPlanner()
        self.navigator = Navigator()
        self.battler = RuleBasedBattler()
        self.manager = ResourceManager()
        self.controller = MenuController()

        # Agent state
        self.mode: str = "explore"
        self.prev_map: int = 0
        self.prev_x: int = 0
        self.prev_y: int = 0

        # Extend the observation space with agent-specific observations
        assert isinstance(env.observation_space, spaces.Dict)
        new_spaces = dict(env.observation_space.spaces)
        new_spaces["battle_state"] = spaces.Box(
            low=-1.0, high=2.0, shape=(64,), dtype=np.float32,
        )
        new_spaces["goal_encoding"] = spaces.Box(
            low=0.0, high=1.0, shape=(16,), dtype=np.float32,
        )
        new_spaces["nav_hint"] = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32,
        )
        self.observation_space = spaces.Dict(new_spaces)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        """
        Reset the environment and all agent modules.

        Returns:
            Tuple of (augmented_observation, info).
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset all agent modules
        self.planner.reset()
        self.navigator.reset()
        self.battler.reset()
        self.manager.reset()
        self.controller.reset()
        self.mode = "explore"

        # Initialize position tracking
        pyboy = self.unwrapped.pyboy  # type: ignore[attr-defined]
        map_id, x, y = memory.get_position(pyboy)
        self.prev_map = map_id
        self.prev_x = x
        self.prev_y = y

        # Initial planner update
        badges = memory.get_badges(pyboy)
        event_count = int(info.get("events", 0))
        self.planner.update(badges, event_count, map_id)

        # Set initial navigation goal
        goal = self.planner.get_current_goal()
        self.navigator.set_goal(goal.target_map_id)

        # Augment observation
        obs = self._augment_obs(obs, pyboy)

        return obs, info

    def step(
        self, action: int,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        """
        Execute one step with agent action routing and reward shaping.

        Step flow:
            1. Detect mode from memory state
            2. If controller has pending sequence → use it
            3. Else if BATTLE → battler decides, controller queues
            4. Else if DIALOGUE → press A
            5. Else EXPLORE → use RL policy's action
            6. Call base env step with actual action
            7. Update planner/manager/navigator
            8. Calculate reward shaping
            9. Augment observation and info

        Args:
            action: RL policy's action (0-7). May be overridden.

        Returns:
            Tuple of (obs, reward + nav_bonus, terminated, truncated, info).
        """
        pyboy = self.unwrapped.pyboy  # type: ignore[attr-defined]

        # =====================================================================
        # Step 1: Detect game mode
        # =====================================================================
        in_battle = memory.is_in_battle(pyboy)
        text_displayed = memory.is_text_displayed(pyboy)

        if in_battle:
            self.mode = "battle"
        elif text_displayed:
            self.mode = "dialogue"
        else:
            self.mode = "explore"

        # =====================================================================
        # Step 2-5: Determine actual action
        # =====================================================================
        actual_action = action  # Default: RL policy's action

        if self.controller.has_pending_sequence():
            # Step 2: Execute pending button sequence
            actual_action = self.controller.get_next_action()

        elif self.mode == "battle" and self.config.agent_battle_control:
            # Step 3: Battler decides, controller queues sequence
            battle_state = self._read_battle_state(pyboy)
            battle_action = self.battler.decide_action(battle_state)
            self.controller.start_from_battle_action(battle_action)
            actual_action = self.controller.get_next_action()

        elif self.mode == "dialogue" and self.config.agent_dialogue_control:
            # Step 4: Press A to advance dialogue
            actual_action = 5  # A button

        # Step 5: EXPLORE mode uses RL policy's action (no override)

        # =====================================================================
        # Step 6: Execute action in base environment
        # =====================================================================
        obs, reward, terminated, truncated, info = self.env.step(actual_action)

        # =====================================================================
        # Step 7: Update agent modules
        # =====================================================================
        map_id, x, y = memory.get_position(pyboy)
        badges = memory.get_badges(pyboy)
        event_count = int(info.get("events", 0))

        # Update planner FSM
        self.planner.update(badges, event_count, map_id)

        # Check manager (heal/grind/continue)
        goal = self.planner.get_current_goal()
        decision = self.manager.evaluate(pyboy, goal, badges)

        # Set navigator goal based on manager decision
        if decision.action == "heal":
            # Navigate to nearest Pokemon Center
            center_map = self.navigator.find_nearest_pokemon_center(map_id)
            if center_map >= 0:
                self.navigator.set_goal(center_map)
            else:
                self.navigator.set_goal(goal.target_map_id)
        elif decision.action == "grind":
            # Stay in current area (no specific goal)
            self.navigator.set_goal(-1)
        else:
            # Continue toward quest goal
            self.navigator.set_goal(goal.target_map_id)

        # =====================================================================
        # Step 8: Calculate navigation reward shaping
        # =====================================================================
        nav_bonus = 0.0
        if self.mode == "explore" and self.config.agent_nav_reward_scale > 0:
            raw_shaping = self.navigator.get_reward_shaping(
                self.prev_map, self.prev_x, self.prev_y,
                map_id, x, y,
                gamma=0.995,
            )
            nav_bonus = raw_shaping * self.config.agent_nav_reward_scale

        # Update position tracking
        self.prev_map = map_id
        self.prev_x = x
        self.prev_y = y

        # =====================================================================
        # Step 9: Augment observation and info
        # =====================================================================
        obs = self._augment_obs(obs, pyboy)

        info["agent_mode"] = self.mode
        info["agent_goal"] = goal.name
        info["agent_nav_bonus"] = nav_bonus
        info["agent_manager_action"] = decision.action

        return obs, float(reward) + nav_bonus, terminated, truncated, info

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _augment_obs(
        self,
        obs: dict[str, np.ndarray[Any, Any]],
        pyboy: object,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """
        Add agent-specific observations to the base observation dict.

        Adds:
        - battle_state: 64-dim battle encoding (zeros when not in battle)
        - goal_encoding: 16-dim quest goal encoding
        - nav_hint: 8-dim navigation direction/distance

        Args:
            obs: Base observation dict from environment.
            pyboy: PyBoy emulator instance.

        Returns:
            Augmented observation dict with 3 new keys.
        """
        # Battle state encoding
        if memory.is_in_battle(pyboy):
            battle_state = self._read_battle_state(pyboy)
            obs["battle_state"] = RuleBasedBattler.get_battle_observation(
                battle_state
            )
        else:
            obs["battle_state"] = np.zeros(64, dtype=np.float32)

        # Goal encoding from planner
        obs["goal_encoding"] = self.planner.get_goal_encoding()

        # Navigation hint
        map_id = memory.read_byte(pyboy, memory.ADDR_MAP_ID)
        obs["nav_hint"] = self.navigator.get_nav_encoding(map_id)

        return obs

    def _read_battle_state(self, pyboy: object) -> BattleState:
        """
        Read full battle state from memory and construct BattleState.

        Args:
            pyboy: PyBoy emulator instance.

        Returns:
            BattleState dataclass with all battle information.
        """
        player_data = memory.get_battle_player_pokemon(pyboy)
        enemy_data = memory.get_battle_enemy_pokemon(pyboy)
        player_mods = memory.get_stat_modifiers(pyboy, is_player=True)
        enemy_mods = memory.get_stat_modifiers(pyboy, is_player=False)
        battle_type = memory.get_battle_type(pyboy)

        # Get party HP fraction
        current_hp, max_hp = memory.get_total_party_hp(pyboy)
        party_hp_fraction = current_hp / max(max_hp, 1)

        return BattleState.from_memory(
            player_data=player_data,
            enemy_data=enemy_data,
            player_mods=player_mods,
            enemy_mods=enemy_mods,
            battle_type=battle_type,
            party_hp_fraction=party_hp_fraction,
        )
