"""
Training script for KantoRL.

This module provides the main training loop for training RL agents on Pokemon Red
using Proximal Policy Optimization (PPO) from stable-baselines3. It handles:
- Creating vectorized parallel environments for faster training
- Configuring the PPO algorithm with tuned hyperparameters
- Auto-resuming from checkpoints for long training runs
- Setting up callbacks for monitoring and logging
- CLI interface for easy training configuration

Architecture Role:
    This is the entry point for training. It orchestrates:
    1. Environment creation (env.py → make_env)
    2. Model configuration (PPO with MultiInputPolicy)
    3. Training callbacks (callbacks.py)
    4. Checkpoint management (auto-resume, periodic saves)

    The flow is:
    CLI args → train() → SubprocVecEnv → PPO → model.learn() → checkpoints

Training Strategy:
    PPO (Proximal Policy Optimization) is used because:
    - Stable and reliable across diverse environments
    - Works well with pixel-based observations
    - Handles discrete action spaces naturally
    - Good sample efficiency for continuous training

    The default hyperparameters match PokemonRedExperiments V2:
    - n_steps=128: Rollout length (steps before update)
    - batch_size=512: Mini-batch size for SGD updates
    - n_epochs=3: Number of optimization passes per rollout
    - gamma=0.998: High discount factor (long-horizon planning)
    - learning_rate=2.5e-4: Standard PPO learning rate

Parallel Environments:
    Training uses SubprocVecEnv to run multiple game instances in parallel:
    - Each environment runs in a separate process
    - Default 16 environments provides 16x throughput
    - More environments = faster training but more RAM

    All environments can optionally stream to a visualization server for
    monitoring training progress in real-time, each with a unique username
    and auto-generated color.

Auto-Resume:
    Training automatically resumes from the latest checkpoint unless
    --no-resume is specified. This allows:
    - Recovering from crashes or interruptions
    - Extending training beyond initial timestep goal
    - Switching machines during long training runs

    Checkpoints include the full model state (policy, optimizer, etc.)

Usage:
    # Basic training (16 envs, 10M steps)
    kantorl train pokemon_red.gb

    # Custom configuration
    kantorl train pokemon_red.gb --envs 8 --steps 5000000 --session my_run

    # Start fresh (ignore checkpoints)
    kantorl train pokemon_red.gb --no-resume

    # With streaming visualization
    kantorl train pokemon_red.gb --stream --stream-user "trainer1"

Dependencies:
    - torch: Neural network backend (PyTorch)
    - stable_baselines3: PPO implementation and vectorized environments
    - kantorl.config: Configuration dataclass
    - kantorl.env: Environment factory function
    - kantorl.callbacks: Training callbacks

References:
    - PPO paper: https://arxiv.org/abs/1707.06347
    - stable-baselines3: https://stable-baselines3.readthedocs.io/
    - PokemonRedExperiments: https://github.com/PWhiddy/PokemonRedExperiments
"""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv

from kantorl.callbacks import (
    CumulativeCheckpointCallback,
    PerformanceCallback,
    StallDetectionCallback,
    TensorboardCallback,
)
from kantorl.config import KantoConfig
from kantorl.env import make_env


# =============================================================================
# POLICY NETWORK CONFIGURATION
# =============================================================================


def create_policy_kwargs() -> dict:
    """
    Create the policy network architecture configuration.

    Defines the neural network structure for both the policy (actor) and
    value (critic) networks. The architecture follows a simple but effective
    design with separate networks for policy and value estimation.

    Network Architecture:
        - Policy network (pi): Input → 256 → 256 → Actions
        - Value network (vf): Input → 256 → 256 → Value (scalar)

        Both networks use ReLU activation functions between layers.
        The input layer processes the feature extractor output (from CNN
        for screens + linear for auxiliary features).

    Why 256-256?
        - 256 units is a common choice that balances capacity and speed
        - Two hidden layers provide enough depth for complex policies
        - Larger networks may overfit, smaller may underfit
        - These values match PokemonRedExperiments V2 architecture

    Why Separate Networks?
        - Policy and value have different learning dynamics
        - Separate networks prevent value updates from destabilizing policy
        - Allows for different network sizes if needed (not used here)

    Returns:
        Dictionary with:
        - net_arch: Defines hidden layer sizes for policy and value networks
        - activation_fn: Activation function class (torch.nn.ReLU)

    Example:
        >>> policy_kwargs = create_policy_kwargs()
        >>> model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)

    Notes:
        - This is passed to PPO's policy_kwargs parameter
        - stable-baselines3 handles the feature extractor automatically
        - MultiInputPolicy uses NatureCNN for image observations
    """
    return {
        # Network architecture specification
        # "pi" = policy network layers, "vf" = value function network layers
        "net_arch": {
            "pi": [256, 256],  # Policy network: 2 hidden layers of 256 units
            "vf": [256, 256],  # Value network: 2 hidden layers of 256 units
        },
        # Activation function between layers
        # ReLU is standard, fast, and works well for most RL tasks
        "activation_fn": torch.nn.ReLU,
    }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def train(
    rom_path: str,
    session_path: str = "runs",
    total_timesteps: int = 10_000_000,
    n_envs: int = 16,
    resume: bool = True,
    reward_fn: str = "default",
    seed: int = 42,
    enable_streaming: bool = False,
    stream_username: str = "kantorl-agent",
    stream_color: str = "#ff0000",
    stream_sprite_id: int = 0,
) -> None:
    """
    Run PPO training on Pokemon Red.

    This is the main training function that sets up the environment, model,
    and callbacks, then runs the training loop. It handles both fresh starts
    and resuming from checkpoints.

    Training Pipeline:
        1. Create configuration object with all settings
        2. Create vectorized environment (multiple parallel games)
        3. Check for existing checkpoints if resume=True
        4. Create or load PPO model
        5. Setup training callbacks (checkpointing, logging, etc.)
        6. Run training loop with progress bar
        7. Save final model on completion or interrupt

    Args:
        rom_path: Path to the Pokemon Red ROM file (.gb file).
                 Must be a valid, unmodified US Pokemon Red ROM.
        session_path: Directory for all training outputs. Default "runs".
                     Creates subdirectories:
                     - session_path/checkpoints/: Model checkpoints
                     - session_path/tensorboard/: TensorBoard logs
        total_timesteps: Total number of environment steps to train.
                        Default 10 million. Each step is one frame of gameplay.
                        Typical training needs 5-50M steps for good results.
        n_envs: Number of parallel environments. Default 16.
               More environments = faster training but more RAM.
               Each environment uses ~100-200MB RAM.
        resume: Whether to auto-resume from the latest checkpoint. Default True.
               Set to False to start fresh training even if checkpoints exist.
        reward_fn: Name of the reward function to use. Default "default".
                  Options: "default", "badges_only", "exploration"
                  See rewards.py for details on each reward function.
        seed: Random seed for reproducibility. Default 42.
             Affects environment randomization and model initialization.
        enable_streaming: Enable real-time streaming to visualization server.
                         Default False. All parallel environments stream.
        stream_username: Display name for the streaming visualization.
        stream_color: Hex color code for the stream display (e.g., "#0033ff").
        stream_sprite_id: Sprite ID for the stream visualization (0-50).

    Side Effects:
        - Creates session_path directory if it doesn't exist
        - Saves checkpoint files to session_path/checkpoints/
        - Writes TensorBoard logs to session_path/tensorboard/
        - Prints progress messages to stdout

    Example:
        >>> # Basic training
        >>> train("pokemon_red.gb")

        >>> # Custom configuration
        >>> train(
        ...     rom_path="pokemon_red.gb",
        ...     session_path="experiments/run1",
        ...     total_timesteps=5_000_000,
        ...     n_envs=8,
        ... )

    Notes:
        - Training can be interrupted with Ctrl+C (saves checkpoint)
        - Checkpoints are saved every 100,000 steps
        - TensorBoard logs are updated every 1,000 steps
        - Stall detection warns if no progress for 50,000 steps
    """
    # Convert session path to Path object for consistent handling
    session_path = Path(session_path)

    # Create session directory and subdirectories
    # parents=True creates intermediate directories, exist_ok=True ignores if exists
    session_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Configuration Setup
    # -------------------------------------------------------------------------
    # Create configuration object with all settings
    # This bundles hyperparameters and paths for easy passing to components
    config = KantoConfig(
        rom_path=rom_path,
        session_path=session_path,
        n_envs=n_envs,
        enable_streaming=enable_streaming,
        stream_username=stream_username,
        stream_color=stream_color,
        stream_sprite_id=stream_sprite_id,
    )

    print(f"Creating {n_envs} parallel environments...")

    # -------------------------------------------------------------------------
    # Environment Creation
    # -------------------------------------------------------------------------
    # Create vectorized environment with multiple parallel instances
    # SubprocVecEnv runs each environment in a separate process for true parallelism

    # Create a list of environment factory functions
    # Each function, when called, creates one environment instance
    # All environments stream if streaming is enabled (each with unique color/name)
    env_fns = [
        make_env(
            rom_path,
            config,
            rank=i,  # Unique ID for this environment instance
            seed=seed,  # Base seed (each env adds rank for unique seeds)
            reward_fn=reward_fn,
            enable_streaming=enable_streaming,  # All environments stream
        )
        for i in range(n_envs)
    ]

    # Create the vectorized environment
    # SubprocVecEnv spawns n_envs processes, each running one environment
    # This provides linear speedup for training (16 envs ≈ 16x faster)
    env = SubprocVecEnv(env_fns)

    # -------------------------------------------------------------------------
    # Checkpoint Detection
    # -------------------------------------------------------------------------
    # Check for existing checkpoints to enable auto-resume
    # This allows training to continue after crashes or manual stops
    checkpoint_path, start_steps = CumulativeCheckpointCallback.find_latest(
        session_path / "checkpoints"
    )

    # -------------------------------------------------------------------------
    # Model Creation or Loading
    # -------------------------------------------------------------------------
    if checkpoint_path and resume:
        # Resume from existing checkpoint
        print(f"Resuming from checkpoint: {checkpoint_path} (step {start_steps:,})")

        # Load the model with all its state (weights, optimizer, etc.)
        # env= parameter updates the model's environment reference
        model = PPO.load(checkpoint_path, env=env)

        # Calculate remaining steps (may be 0 if already completed)
        remaining_steps = max(0, total_timesteps - start_steps)
    else:
        # Start fresh training with new model
        print("Starting fresh training...")

        # Create new PPO model with configured hyperparameters
        model = PPO(
            # Policy type: MultiInputPolicy handles Dict observation spaces
            # It uses NatureCNN for image observations (screens) and
            # MLP for vector observations (health, badges, etc.)
            "MultiInputPolicy",
            env,
            # Rollout parameters
            n_steps=config.n_steps,  # Steps per env before update (128)
            batch_size=config.batch_size,  # Mini-batch size for SGD (512)
            n_epochs=config.n_epochs,  # Optimization passes per rollout (3)
            # Discount and advantage estimation
            gamma=config.gamma,  # Discount factor (0.998 = long horizon)
            gae_lambda=config.gae_lambda,  # GAE lambda (0.95)
            # PPO clipping
            clip_range=config.clip_range,  # Policy clip range (0.2)
            # Loss coefficients
            ent_coef=config.ent_coef,  # Entropy bonus (0.01 = mild exploration)
            vf_coef=config.vf_coef,  # Value function coefficient (0.5)
            # Optimizer
            learning_rate=config.learning_rate,  # Learning rate (2.5e-4)
            # Network architecture (defined above)
            policy_kwargs=create_policy_kwargs(),
            # Logging
            tensorboard_log=str(session_path / "tensorboard"),
            verbose=1,  # Print training info
            # Reproducibility
            seed=seed,
        )
        remaining_steps = total_timesteps

    # -------------------------------------------------------------------------
    # Callback Setup
    # -------------------------------------------------------------------------
    # Create list of callbacks that execute during training
    # Callbacks provide hooks for checkpointing, logging, and monitoring
    callbacks = CallbackList([
        # Save model checkpoints periodically
        # Saves model_100000.zip, model_200000.zip, etc.
        CumulativeCheckpointCallback(
            save_path=session_path / "checkpoints",
            save_freq=100_000,  # Save every 100K steps
        ),
        # Log game-specific metrics to TensorBoard
        # Tracks badges, events, exploration progress
        TensorboardCallback(log_freq=1000),
        # Warn if training appears stalled
        # Detects when reward stops improving
        StallDetectionCallback(check_freq=50_000),
        # Log training speed (steps/second)
        # Helps identify performance issues
        PerformanceCallback(log_freq=10_000),
    ])

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print(f"Training for {remaining_steps:,} steps...")
    print(f"Tensorboard: tensorboard --logdir {session_path / 'tensorboard'}")

    try:
        # Run the main training loop
        model.learn(
            total_timesteps=remaining_steps,
            callback=callbacks,
            # Don't reset step counter when resuming
            # This ensures checkpoint names continue from where we left off
            reset_num_timesteps=False,
            # Show progress bar for visual feedback
            progress_bar=True,
        )
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nTraining interrupted. Saving checkpoint...")
    finally:
        # -------------------------------------------------------------------------
        # Cleanup and Final Save
        # -------------------------------------------------------------------------
        # Always save a final checkpoint, even on interrupt
        # This prevents losing progress from the last save interval
        final_path = session_path / "checkpoints" / f"model_{model.num_timesteps}.zip"
        model.save(final_path)
        print(f"Saved final model: {final_path}")

        # Close the vectorized environment
        # This properly shuts down all subprocess workers
        env.close()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


def main() -> None:
    """
    CLI entry point for training.

    Parses command-line arguments and calls the train() function.
    This is invoked by the 'kantorl train' command defined in pyproject.toml.

    Arguments:
        rom_path: Required. Path to the Pokemon Red ROM file.
        --session, -s: Session directory for outputs. Default: "runs"
        --steps, -n: Total training steps. Default: 10,000,000
        --envs, -e: Number of parallel environments. Default: 16
        --no-resume: Start fresh, ignoring checkpoints.
        --reward, -r: Reward function name. Default: "default"
        --seed: Random seed. Default: 42
        --stream: Enable streaming visualization.
        --stream-user: Username for stream display.
        --stream-color: Hex color for stream display.
        --stream-sprite: Sprite ID for stream display.

    Example Usage:
        $ kantorl train pokemon_red.gb
        $ kantorl train pokemon_red.gb --envs 8 --steps 5000000
        $ kantorl train pokemon_red.gb --no-resume --reward exploration
        $ kantorl train pokemon_red.gb --stream --stream-user "trainer1"
    """
    # Create argument parser with description
    parser = argparse.ArgumentParser(description="Train PPO on Pokemon Red")

    # -------------------------------------------------------------------------
    # Required Arguments
    # -------------------------------------------------------------------------
    parser.add_argument(
        "rom_path",
        help="Path to Pokemon Red ROM file (.gb)",
    )

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--session", "-s",
        default="runs",
        help="Session directory for checkpoints and logs (default: runs)",
    )
    parser.add_argument(
        "--steps", "-n",
        type=int,
        default=10_000_000,
        help="Total training steps (default: 10,000,000)",
    )
    parser.add_argument(
        "--envs", "-e",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing checkpoint (start fresh)",
    )
    parser.add_argument(
        "--reward", "-r",
        default="default",
        choices=["default", "badges_only", "exploration"],
        help="Reward function to use (default: default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # -------------------------------------------------------------------------
    # Streaming Options
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming to shared map visualization server",
    )
    parser.add_argument(
        "--stream-user",
        default="kantorl-agent",
        help="Username for stream display (default: kantorl-agent)",
    )
    parser.add_argument(
        "--stream-color",
        default="#ff0000",
        help="Hex color for stream display (default: #ff0000)",
    )
    parser.add_argument(
        "--stream-sprite",
        type=int,
        default=0,
        help="Sprite ID for stream display, 0-50 (default: 0)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Call train function with parsed arguments
    train(
        rom_path=args.rom_path,
        session_path=args.session,
        total_timesteps=args.steps,
        n_envs=args.envs,
        resume=not args.no_resume,  # --no-resume flag inverts resume=True default
        reward_fn=args.reward,
        seed=args.seed,
        enable_streaming=args.stream,
        stream_username=args.stream_user,
        stream_color=args.stream_color,
        stream_sprite_id=args.stream_sprite,
    )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Allow running directly with: python -m kantorl.train
if __name__ == "__main__":
    main()
