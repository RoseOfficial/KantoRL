"""
Evaluation script for KantoRL.

This module provides functionality to evaluate trained RL agents on Pokemon Red.
Evaluation runs the agent for multiple episodes and collects performance metrics
to assess how well the agent has learned to play the game.

Architecture Role:
    The eval module is used after training to assess agent quality:
    1. Load a trained model checkpoint
    2. Run the agent for multiple episodes
    3. Collect metrics (rewards, badges, exploration)
    4. Generate summary statistics

    Unlike training, evaluation uses deterministic actions (no exploration)
    and runs in a single environment (no parallelism needed).

Why Evaluate?
    Training metrics (loss, entropy) don't always correlate with performance.
    Evaluation provides direct measures of:
    - Game progress (badges collected, events triggered)
    - Exploration ability (unique tiles visited)
    - Reward accumulation (total episode reward)
    - Consistency (variance across episodes)

Evaluation vs Training:
    - Evaluation uses deterministic action selection (no random exploration)
    - Evaluation runs serially (one environment, easier to debug)
    - Evaluation can render the game for visual inspection
    - Evaluation focuses on metrics, not learning

Usage:
    # From command line
    kantorl eval runs/checkpoints/model_1000000.zip pokemon_red.gb

    # With rendering (watch the agent play)
    kantorl eval runs/checkpoints/model_1000000.zip pokemon_red.gb --render

    # More episodes for better statistics
    kantorl eval checkpoint.zip rom.gb --episodes 50

    # From Python
    from kantorl.eval import evaluate
    results = evaluate("checkpoint.zip", "pokemon_red.gb", n_episodes=10)
    print(f"Max badges achieved: {results['max_badges']}")

Metrics Collected:
    - episode_rewards: Total reward per episode
    - episode_lengths: Number of steps per episode
    - final_badges: Badges at episode end
    - final_events: Event flags at episode end
    - max_explore_tiles: Peak exploration tiles visited

Dependencies:
    - numpy: For metric aggregation
    - stable_baselines3: For loading PPO models
    - kantorl.config: Configuration dataclass
    - kantorl.env: Environment for running the agent
"""

import numpy as np
from stable_baselines3 import PPO

from kantorl.config import KantoConfig
from kantorl.env import KantoRedEnv


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================


def evaluate(
    checkpoint_path: str,
    rom_path: str,
    n_episodes: int = 10,
    max_steps: int = 50_000,
    render: bool = False,
) -> dict:
    """
    Evaluate a trained agent on Pokemon Red.

    Loads a trained model checkpoint and runs it for multiple episodes,
    collecting performance metrics. The agent uses deterministic actions
    (no exploration noise) to assess true learned behavior.

    Evaluation Process:
        1. Load the PPO model from checkpoint
        2. Create a single evaluation environment
        3. For each episode:
           a. Reset environment to starting state
           b. Run agent until episode ends (max_steps or game over)
           c. Record metrics (reward, badges, events, exploration)
        4. Aggregate and report statistics

    Args:
        checkpoint_path: Path to the model checkpoint file (.zip).
                        This is the file saved by CumulativeCheckpointCallback
                        during training (e.g., "runs/checkpoints/model_1000000.zip").
        rom_path: Path to the Pokemon Red ROM file (.gb).
                 Must be the same ROM used during training.
        n_episodes: Number of evaluation episodes to run. Default 10.
                   More episodes = more reliable statistics but longer runtime.
                   Recommended: 10 for quick checks, 50+ for final evaluation.
        max_steps: Maximum steps per episode before truncation. Default 50,000.
                  This prevents infinite loops if the agent gets stuck.
                  50,000 steps ≈ 14 hours of real-time gameplay at 60fps.
        render: Whether to render the gameplay visually. Default False.
               When True, opens a window showing the GameBoy screen.
               Useful for debugging and understanding agent behavior.
               Note: Rendering significantly slows down evaluation.

    Returns:
        Dictionary containing aggregated evaluation metrics:
        - episodes: Number of episodes evaluated
        - mean_reward: Average total reward per episode
        - std_reward: Standard deviation of rewards (consistency measure)
        - mean_length: Average episode length in steps
        - max_badges: Maximum badges achieved in any episode
        - mean_badges: Average badges at episode end
        - mean_events: Average event flags triggered
        - mean_explore_tiles: Average peak exploration tiles visited

    Example:
        >>> # Quick evaluation
        >>> results = evaluate("model_1000000.zip", "pokemon_red.gb")
        >>> print(f"Agent achieved {results['max_badges']} badges")

        >>> # Detailed evaluation with rendering
        >>> results = evaluate(
        ...     checkpoint_path="model_5000000.zip",
        ...     rom_path="pokemon_red.gb",
        ...     n_episodes=50,
        ...     render=True,
        ... )

    Notes:
        - Uses deterministic=True for action selection (no randomness)
        - Metrics are collected from info dict returned by env.step()
        - Final metrics come from the last step before episode end
        - Rendering requires a display; won't work on headless servers
    """
    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------
    # Load the trained PPO model from the checkpoint file
    # The checkpoint contains the policy network weights, optimizer state,
    # and training statistics (though we only need the policy for evaluation)
    model = PPO.load(checkpoint_path)

    # -------------------------------------------------------------------------
    # Environment Setup
    # -------------------------------------------------------------------------
    # Create configuration with evaluation-specific settings
    config = KantoConfig(
        rom_path=rom_path,
        max_steps=max_steps,  # Override default to use evaluation max_steps
    )

    # Set render mode based on user preference
    # "human" mode opens a PyGame window showing the GameBoy screen
    # None mode runs headless (much faster, no visual output)
    render_mode = "human" if render else None

    # Create a single evaluation environment (no parallelism needed)
    env = KantoRedEnv(config=config, render_mode=render_mode)

    # -------------------------------------------------------------------------
    # Metric Collection Setup
    # -------------------------------------------------------------------------
    # Lists to store per-episode metrics
    episode_rewards = []  # Total reward accumulated in each episode
    episode_lengths = []  # Number of steps in each episode
    final_badges = []  # Badges at episode end (progress metric)
    final_events = []  # Event flags at episode end (story progress)
    max_explore_tiles = []  # Peak exploration tiles (exploration ability)

    print(f"Evaluating {checkpoint_path} for {n_episodes} episodes...")

    # -------------------------------------------------------------------------
    # Evaluation Loop
    # -------------------------------------------------------------------------
    for ep in range(n_episodes):
        # Reset environment to get initial observation
        # The observation is a Dict with screens, health, etc.
        obs, info = env.reset()

        # Episode tracking variables
        done = False  # Episode end flag
        ep_reward = 0.0  # Cumulative reward this episode
        ep_length = 0  # Steps taken this episode
        max_tiles = 0  # Peak exploration tiles seen

        # Run episode until terminated or truncated
        while not done:
            # Get action from the trained policy
            # deterministic=True means we use the argmax action (no sampling)
            # This gives us the agent's "best guess" action, not exploration
            action, _ = model.predict(obs, deterministic=True)

            # Execute action in environment
            # Returns: new observation, reward, terminated flag, truncated flag, info dict
            obs, reward, terminated, truncated, info = env.step(action)

            # Episode ends if terminated (game over) or truncated (max steps)
            done = terminated or truncated

            # Update episode metrics
            ep_reward += reward
            ep_length += 1

            # Track peak exploration (may fluctuate as agent revisits areas)
            max_tiles = max(max_tiles, info.get("explore_tiles", 0))

        # -------------------------------------------------------------------------
        # Episode Complete - Record Metrics
        # -------------------------------------------------------------------------
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        # Extract final game state from info dict
        # These come from the last step of the episode
        final_badges.append(info.get("badges", 0))
        final_events.append(info.get("events", 0))
        max_explore_tiles.append(max_tiles)

        # Print per-episode summary
        print(
            f"Episode {ep + 1}: Reward={ep_reward:.1f}, "
            f"Badges={info.get('badges', 0)}, "
            f"Events={info.get('events', 0)}, "
            f"Tiles={max_tiles}"
        )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    # Close the environment to properly shut down PyBoy
    env.close()

    # -------------------------------------------------------------------------
    # Aggregate Statistics
    # -------------------------------------------------------------------------
    # Compute summary statistics across all episodes
    results = {
        # Episode count
        "episodes": n_episodes,
        # Reward statistics
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),  # Consistency measure
        # Episode length
        "mean_length": np.mean(episode_lengths),
        # Badge metrics (key progress indicator)
        "max_badges": max(final_badges),  # Best performance
        "mean_badges": np.mean(final_badges),  # Average performance
        # Story progress
        "mean_events": np.mean(final_events),
        # Exploration ability
        "mean_explore_tiles": np.mean(max_explore_tiles),
    }

    # -------------------------------------------------------------------------
    # Print Summary Report
    # -------------------------------------------------------------------------
    print("\n=== Evaluation Results ===")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length: {results['mean_length']:.0f} steps")
    print(f"Max Badges:  {results['max_badges']}")
    print(f"Mean Badges: {results['mean_badges']:.2f}")
    print(f"Mean Events: {results['mean_events']:.0f}")
    print(f"Mean Tiles:  {results['mean_explore_tiles']:.0f}")

    return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

# Allow running directly with: python -m kantorl.eval
if __name__ == "__main__":
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate a trained KantoRL agent on Pokemon Red"
    )

    # Required arguments
    parser.add_argument(
        "checkpoint",
        help="Path to model checkpoint file (.zip)",
    )
    parser.add_argument(
        "rom_path",
        help="Path to Pokemon Red ROM file (.gb)",
    )

    # Optional arguments
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render gameplay (opens visualization window)",
    )

    # Parse arguments and run evaluation
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        rom_path=args.rom_path,
        n_episodes=args.episodes,
        render=args.render,
    )
