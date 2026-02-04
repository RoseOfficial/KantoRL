"""
Analyze if the agent is actually learning across checkpoints.

This utility script helps diagnose whether a training run is making progress
by comparing agent behavior at different checkpoint stages. It's useful for:
- Debugging training runs that seem stuck
- Understanding how agent behavior evolves over time
- Identifying when learning plateaus or regresses
- Validating that training is working as expected

How It Works:
    The script loads multiple checkpoints from a training run and runs each
    one for a short episode (500 steps by default). It collects metrics about:
    - Action distribution (which buttons the agent presses)
    - Action entropy (how random/decisive the actions are)
    - Movement exploration (how many unique positions visited)
    - Reward accumulation (total reward in the test episode)

    By comparing these metrics across checkpoints, you can see if the agent
    is developing more decisive strategies (lower entropy), exploring more
    of the game (more unique positions), or getting higher rewards.

Interpreting Results:
    - Decreasing entropy: Agent is becoming more decisive (good sign)
    - Increasing unique positions: Agent is exploring more of the map
    - Changing dominant action: Agent is trying different strategies
    - Higher total reward: Agent is making better decisions

    Note: Learning in Pokemon Red is slow and non-monotonic. Don't expect
    smooth improvement - the agent may seem stuck for millions of steps
    before making breakthroughs.

Usage:
    # Edit the script to point to your checkpoint directory, then run:
    python analyze_learning.py

    # You'll need to modify these lines in main():
    # - checkpoints path: "test_stream_clean/checkpoints/model_*.zip"
    # - rom path: "pokemon_red.gb"
    # - test_points: list of checkpoint steps to analyze

Example Output:
    === LEARNING ANALYSIS: Is the Agent Actually Learning? ===

    Analyzing 100,000 steps...
    Analyzing 500,000 steps...
    ...

    === RESULTS ===

    Action Entropy (randomness - lower = more decisive):
       100,000: 1.823 - Dominant: RIGHT (23.4%)
       500,000: 1.654 - Dominant: UP (28.1%)

    Movement Exploration:
       100,000: 45 unique positions, range (12, 8)
       500,000: 67 unique positions, range (15, 12)

    Reward Accumulation (per 500 steps):
       100,000: 1.23
       500,000: 2.45

    === LEARNING ASSESSMENT ===
    Entropy change: -0.169 (decreasing (good))
    Action diversity: 2 different dominant actions across training
    Exploration trend: 22 position change
    ✅ Agent IS learning - developing consistent strategies

Configuration:
    Before running, modify these variables in main():
    - Checkpoint path pattern (line ~67)
    - ROM path (line ~80)
    - Test points - which checkpoint steps to analyze (line ~70)

Dependencies:
    - numpy: For statistical calculations
    - stable_baselines3: For loading PPO models
    - kantorl: For the environment and memory reading
    - glob: For finding checkpoint files

Notes:
    - This is a standalone utility script, not part of the main package
    - Hardcoded paths need to be updated for your training run
    - Runs with deterministic=False to see natural agent behavior
    - Short episodes (500 steps) may not show full behavior patterns
"""

import glob

import numpy as np
from stable_baselines3 import PPO

from kantorl import KantoRedEnv


# =============================================================================
# CHECKPOINT ANALYSIS
# =============================================================================


def analyze_checkpoint(checkpoint_path: str, rom_path: str, n_steps: int = 500) -> dict:
    """
    Perform quick analysis of a checkpoint's behavior.

    Loads a trained model and runs it for a short episode, collecting metrics
    about its action distribution, movement patterns, and reward accumulation.

    Analysis Metrics:
        - Action entropy: Measures randomness of action selection
          (higher = more random, lower = more decisive/consistent)
        - Dominant action: Which action the agent uses most frequently
        - Unique positions: Number of distinct (x, y) coordinates visited
        - Movement range: How far the agent moves in each direction
        - Total reward: Sum of rewards during the test episode

    Args:
        checkpoint_path: Path to the model checkpoint file (.zip).
        rom_path: Path to the Pokemon Red ROM file.
        n_steps: Number of steps to run for analysis. Default 500.
                More steps = more reliable statistics but slower.

    Returns:
        Dictionary containing analysis metrics:
        - entropy: Action distribution entropy (float)
        - dominant_action: Index of most-used action (int, 0-7)
        - dominant_pct: Percentage of steps using dominant action (float)
        - total_reward: Cumulative reward during episode (float)
        - unique_positions: Number of unique (x, y) positions (int)
        - movement_range: Tuple of (x_range, y_range) movement extent
        - action_distribution: Array of action counts [8,]

    Example:
        >>> results = analyze_checkpoint(
        ...     "runs/checkpoints/model_1000000.zip",
        ...     "pokemon_red.gb",
        ...     n_steps=500
        ... )
        >>> print(f"Entropy: {results['entropy']:.3f}")
        >>> print(f"Dominant action: {results['dominant_action']}")
    """
    # Load the trained model
    model = PPO.load(checkpoint_path)

    # Create environment (headless, no rendering)
    env = KantoRedEnv(rom_path=rom_path, render_mode=None)

    # Reset to get initial observation
    obs, info = env.reset()

    # Initialize metric tracking arrays
    actions = np.zeros(8)  # Count of each action (8 possible actions)
    positions_x = []       # X coordinates visited
    positions_y = []       # Y coordinates visited
    rewards = []           # Rewards received

    # Run the episode
    for _ in range(n_steps):
        # Get action from model (deterministic=False to see natural behavior)
        # Using stochastic actions shows what the policy actually learned
        action, _ = model.predict(obs, deterministic=False)

        # Execute action
        obs, reward, done, trunc, info = env.step(action)

        # Record metrics
        actions[action] += 1
        rewards.append(reward)

        # Try to get position from game memory if available
        # This gives us more detailed movement tracking
        if hasattr(env, 'pyboy'):
            from kantorl import memory
            x = memory.read_byte(env.pyboy, memory.ADDR_PLAYER_X)
            y = memory.read_byte(env.pyboy, memory.ADDR_PLAYER_Y)
            positions_x.append(x)
            positions_y.append(y)

        # End episode if done (and break to avoid reset complications)
        if done or trunc:
            obs, info = env.reset()
            break

    # Clean up
    env.close()

    # -------------------------------------------------------------------------
    # Calculate Derived Metrics
    # -------------------------------------------------------------------------

    # Action entropy: measures randomness of action selection
    # Formula: -sum(p * log(p)) where p is probability of each action
    # Higher entropy = more random, lower = more decisive
    action_probs = actions / n_steps
    action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

    # Dominant action analysis
    dominant_action = np.argmax(actions)
    dominant_pct = np.max(action_probs) * 100

    # Movement analysis
    # Unique positions shows exploration breadth
    unique_positions = len(set(zip(positions_x, positions_y))) if positions_x else 0

    # Movement range shows how far the agent moves in each direction
    movement_range = (
        max(positions_x) - min(positions_x) if positions_x else 0,
        max(positions_y) - min(positions_y) if positions_y else 0
    )

    return {
        'entropy': action_entropy,
        'dominant_action': dominant_action,
        'dominant_pct': dominant_pct,
        'total_reward': sum(rewards),
        'unique_positions': unique_positions,
        'movement_range': movement_range,
        'action_distribution': actions.astype(int)
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================


def main():
    """
    Run learning analysis across multiple checkpoints.

    Loads checkpoints at specified training steps and compares their
    behavior to determine if learning is occurring.

    Configuration (modify these values for your training run):
        - checkpoints: Glob pattern for checkpoint files
        - test_points: List of step counts to analyze
        - rom_path: Path to your Pokemon Red ROM

    Assessment Criteria:
        - Entropy decreasing: Agent is becoming more decisive
        - Action diversity: Agent is trying different strategies
        - Exploration trend: Agent is covering more ground
    """
    print("=== LEARNING ANALYSIS: Is the Agent Actually Learning? ===\n")

    # -------------------------------------------------------------------------
    # CONFIGURATION - MODIFY THESE FOR YOUR TRAINING RUN
    # -------------------------------------------------------------------------

    # Path pattern for checkpoint files
    # Change this to match your training session directory
    checkpoint_pattern = "test_stream_clean/checkpoints/model_*.zip"

    # Path to Pokemon Red ROM
    rom_path = "pokemon_red.gb"

    # Checkpoint steps to analyze
    # Add or remove steps based on your training progress
    test_points = [100000, 500000, 1000000, 1500000, 2000000, 2600000]

    # -------------------------------------------------------------------------
    # FIND AND ANALYZE CHECKPOINTS
    # -------------------------------------------------------------------------

    # Get list of available checkpoints
    checkpoints = sorted(glob.glob(checkpoint_pattern))

    # Human-readable action names for display
    action_names = ["NOOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]

    # Store results for each checkpoint
    results = {}

    # Analyze each target checkpoint
    for target in test_points:
        # Construct expected checkpoint path
        checkpoint = f"test_stream_clean/checkpoints/model_{target}.zip"

        # Only analyze if checkpoint exists
        if checkpoint in checkpoints:
            print(f"Analyzing {target:,} steps...")
            results[target] = analyze_checkpoint(checkpoint, rom_path, n_steps=500)

    # -------------------------------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------------------------------

    print("\n=== RESULTS ===\n")

    # Show entropy progression (decisiveness over time)
    print("Action Entropy (randomness - lower = more decisive):")
    for steps, data in results.items():
        dominant_name = action_names[data['dominant_action']]
        print(f"  {steps:8,}: {data['entropy']:.3f} - "
              f"Dominant: {dominant_name} ({data['dominant_pct']:.1f}%)")

    # Show movement exploration
    print("\nMovement Exploration:")
    for steps, data in results.items():
        print(f"  {steps:8,}: {data['unique_positions']} unique positions, "
              f"range {data['movement_range']}")

    # Show reward accumulation
    print("\nReward Accumulation (per 500 steps):")
    for steps, data in results.items():
        print(f"  {steps:8,}: {data['total_reward']:.2f}")

    # -------------------------------------------------------------------------
    # LEARNING ASSESSMENT
    # -------------------------------------------------------------------------

    print("\n=== LEARNING ASSESSMENT ===")

    # Calculate trends across checkpoints
    entropies = [data['entropy'] for data in results.values()]
    entropy_change = entropies[-1] - entropies[0] if len(entropies) >= 2 else 0

    dominant_actions = [data['dominant_action'] for data in results.values()]
    action_diversity = len(set(dominant_actions))

    positions = [data['unique_positions'] for data in results.values()]
    exploration_trend = positions[-1] - positions[0] if positions and positions[0] > 0 else 0

    # Print trend analysis
    entropy_assessment = 'decreasing (good)' if entropy_change < 0 else 'increasing (exploring)'
    print(f"Entropy change: {entropy_change:.3f} ({entropy_assessment})")
    print(f"Action diversity: {action_diversity} different dominant actions across training")
    print(f"Exploration trend: {exploration_trend} position change")

    # Final assessment
    # Multiple criteria because learning manifests differently at different stages
    if entropy_change < -0.1:
        print("✅ Agent IS learning - developing consistent strategies")
    elif action_diversity > 2:
        print("✅ Agent IS learning - trying different approaches")
    elif exploration_trend > 0:
        print("✅ Agent IS learning - expanding exploration")
    else:
        print("⚠️ Learning is slow but this is normal for Pokemon Red")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
