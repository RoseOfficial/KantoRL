"""
Modular game-aware agent system for KantoRL.

This package provides a hierarchical agent that separates game knowledge from
learned behavior. Hand-authored modules handle what's known (type chart,
quest order, menu navigation), while RL handles what requires adaptive
behavior (overworld exploration with goal guidance).

Architecture:
    ┌──────────────────────────────────────────────┐
    │         STRATEGIC PLANNER (Quest FSM)         │
    │  Hand-authored progression: get_pokedex →     │
    │  route_1 → viridian → pewter → beat_brock    │
    ├────────────┬───────────┬─────────────────────┤
    │ NAVIGATOR  │  BATTLER  │  MANAGER            │
    │ Map graph  │ Rule-based│  Heal/grind          │
    │ + A* path  │ type-aware│  heuristics          │
    │ + reward   │ + scripted│                      │
    │   shaping  │   menus   │                      │
    ├────────────┴───────────┴─────────────────────┤
    │        CONTROLLER (Scripted Sequences)        │
    │  Battle menus, dialogue A-spam, NPC interact  │
    ├──────────────────────────────────────────────┤
    │           AgentWrapper (gym.Wrapper)          │
    │  Mode detection → action routing → obs augment│
    └──────────────────────────────────────────────┘

Modules:
    - planner: Quest FSM encoding Pokemon Red's progression order
    - navigator: Map graph with A* pathfinding and reward shaping
    - battler: Rule-based battle policy using type chart
    - manager: Resource management heuristics (heal/grind decisions)
    - controller: Scripted button sequences for menus and dialogue
    - wrapper: AgentWrapper that integrates all modules

Wrapper Order:
    KantoRedEnv → CurriculumWrapper → AgentWrapper → StreamWrapper

Usage:
    # Enable via CLI flag
    kantorl train pokemon_red.gb --agent --envs 8 --no-resume

    # Or programmatically
    from kantorl.agent import AgentWrapper
    env = AgentWrapper(env, config=cfg)

Dependencies:
    - gymnasium: For the Wrapper base class
    - numpy: For observation augmentation
    - kantorl.memory: For reading game state
    - kantorl.config: For agent configuration fields
"""

from kantorl.agent.wrapper import AgentWrapper

__all__ = ["AgentWrapper"]
