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

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from kantorl.callbacks import (
    CumulativeCheckpointCallback,
    PerformanceCallback,
    StallDetectionCallback,
    TensorboardCallback,
    WandbCallback,
)
from kantorl.config import KantoConfig
from kantorl.env import make_env

if TYPE_CHECKING:
    import wandb


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
# LSTM POLICY CONFIGURATION (CURRICULUM MODE)
# =============================================================================


def create_lstm_policy_kwargs() -> dict:
    """
    Create the LSTM policy network architecture for RecurrentPPO.

    When curriculum learning is enabled, we use RecurrentPPO with LSTM
    layers instead of standard PPO. The LSTM provides temporal memory
    across timesteps, which is critical for:
    - Navigating long sequences (caves, buildings)
    - Remembering what the agent was doing before a battle
    - Maintaining context across the longer episodes in curriculum mode

    Network Architecture:
        - Shared LSTM: 256 hidden units, 1 layer (feeds both policy and critic)
        - Critic LSTM: 256 hidden units, 1 layer (additional critic-only processing)
        - Policy MLP: Shared LSTM output → 256 → Actions
        - Critic MLP: Shared LSTM + Critic LSTM output → 256 → Value

    Why Shared LSTM?
        shared_lstm=True lets value function gradients shape the shared LSTM's
        temporal representations. With separate LSTMs, the critic learns a
        near-perfect value function (explained_variance=0.99) but the policy
        LSTM gets zero gradient signal from the critic — its temporal features
        are trained only by the near-zero policy gradient, causing vanishing
        gradients (clip_fraction=0, approx_kl≈3e-5). Sharing the LSTM gives
        the policy useful features shaped by value learning.

        sb3-contrib requires shared_lstm and enable_critic_lstm to be mutually
        exclusive. With shared_lstm=True, both heads use the same LSTM output
        and their separate MLP heads (pi: [256], vf: [256]) provide
        head-specific processing.

    Returns:
        Dictionary with LSTM and network architecture configuration.

    Example:
        >>> kwargs = create_lstm_policy_kwargs()
        >>> model = RecurrentPPO("MultiInputLstmPolicy", env, policy_kwargs=kwargs)

    Notes:
        BREAKING: Changing shared_lstm invalidates existing RecurrentPPO
        checkpoints. Must use --no-resume and delete old checkpoints.
    """
    return {
        # LSTM configuration
        "lstm_hidden_size": 256,     # Hidden state size for temporal memory
        "n_lstm_layers": 1,          # Single LSTM layer (sufficient, faster)
        "shared_lstm": True,         # Share LSTM so value gradients shape policy features
        "enable_critic_lstm": False,  # Mutually exclusive with shared_lstm in sb3-contrib

        # MLP layers after shared LSTM output
        "net_arch": {
            "pi": [256],  # Policy: Shared LSTM → 256 → Actions
            "vf": [256],  # Value: Shared LSTM → 256 → Value
        },
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
    stream_username: str = "KantoRL",
    stream_color: str = "#ff0000",
    stream_sprite_id: int = 0,
    use_curriculum: bool = False,
    enable_wandb: bool = False,
    wandb_project: str = "kantorl",
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    use_agent: bool = False,
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
        use_curriculum: Enable curriculum learning with auto-checkpointing,
                       HM automation, and LSTM policy (RecurrentPPO).
                       Requires: pip install -e ".[curriculum]"
        enable_wandb: Enable Weights & Biases experiment tracking.
                     Runs alongside TensorBoard, not replacing it.
                     Requires: pip install -e ".[wandb]"
        wandb_project: wandb project name for grouping runs. Default "kantorl".
        wandb_entity: wandb entity (team/username). None uses personal default.
        wandb_group: Optional run group for organizing related experiments.
        use_agent: Enable modular agent system with quest planning, navigation
                  reward shaping, rule-based battle control, and dialogue handling.
                  Requires: --no-resume for first run (observation space changes).

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
        enable_curriculum=use_curriculum,
        enable_agent=use_agent,
        enable_wandb=enable_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
    )

    # -------------------------------------------------------------------------
    # Wandb Initialization (optional)
    # -------------------------------------------------------------------------
    # Initialize wandb before model creation so the full config is captured.
    # The run object is passed to WandbCallback for metric logging.
    wandb_run: wandb.sdk.wandb_run.Run | None = None
    if enable_wandb:
        import wandb as _wandb

        wandb_run = _wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
            config=dataclasses.asdict(config),
            name=session_path.name,  # type: ignore[attr-defined]
            resume="allow",
            save_code=True,
        )
        print(f"Wandb run: {wandb_run.url}")

    print(f"Creating {n_envs} parallel environments...")

    # -------------------------------------------------------------------------
    # Environment Creation
    # -------------------------------------------------------------------------
    # Create vectorized environment with multiple parallel instances
    # SubprocVecEnv runs each environment in a separate process for true parallelism

    # Create a list of environment factory functions
    # Each function, when called, creates one environment instance
    # All environments stream if streaming is enabled (each with unique color/name)
    # Stagger offsets prevent synchronized episode truncation across all envs
    stagger_step = config.max_steps // n_envs

    env_fns = [
        make_env(
            rom_path,
            config,
            rank=i,  # Unique ID for this environment instance
            seed=seed,  # Base seed (each env adds rank for unique seeds)
            reward_fn=reward_fn,
            enable_streaming=enable_streaming,  # All environments stream
            enable_curriculum=use_curriculum,   # Curriculum wrapping
            enable_agent=use_agent,             # Modular agent system
            initial_step_offset=i * stagger_step,  # Stagger first truncation
        )
        for i in range(n_envs)
    ]

    # Create the vectorized environment
    # SubprocVecEnv spawns n_envs processes, each running one environment
    # This provides linear speedup for training (16 envs ≈ 16x faster)
    raw_env = SubprocVecEnv(env_fns)

    # -------------------------------------------------------------------------
    # Checkpoint Detection
    # -------------------------------------------------------------------------
    # Check for existing checkpoints to enable auto-resume
    # This allows training to continue after crashes or manual stops
    checkpoint_path, start_steps = CumulativeCheckpointCallback.find_latest(
        session_path / "checkpoints"
    )

    # -------------------------------------------------------------------------
    # Reward & Observation Normalization (VecNormalize)
    # -------------------------------------------------------------------------
    # VecNormalize maintains running mean/variance of observations and rewards,
    # normalizing them to roughly unit Gaussian. This is critical because:
    # - Raw exploration rewards are ~0.005/step, producing near-zero advantages
    # - Without normalization, the value function predicts these perfectly,
    #   making policy gradients vanish (clip_fraction=0, approx_kl≈1e-7)
    # - VecNormalize rescales rewards to unit variance, giving PPO meaningful
    #   gradient signal regardless of the raw reward magnitude
    vecnorm_path = session_path / "checkpoints" / "vecnormalize.pkl"

    if checkpoint_path and resume and vecnorm_path.exists():
        # Restore normalization statistics from a previous training session
        # training=True keeps updating the running stats during training
        env = VecNormalize.load(str(vecnorm_path), raw_env)
        env.training = True
    else:
        # Fresh normalization — stats will converge within the first few updates
        # norm_obs=False because our Dict obs contains MultiBinary spaces
        # (badges, events) which VecNormalize can't normalize. The continuous
        # obs (screens, health, level) are already bounded, so reward
        # normalization alone is the critical fix for vanishing gradients.
        env = VecNormalize(
            raw_env,
            norm_obs=False,      # Dict obs has MultiBinary — skip obs norm
            norm_reward=True,    # Normalize rewards to unit variance
            clip_reward=10.0,    # Clip normalized rewards to [-10, 10]
        )

    # -------------------------------------------------------------------------
    # Model Creation or Loading
    # -------------------------------------------------------------------------

    # Detect whether to use RecurrentPPO (LSTM) or standard PPO
    # Curriculum mode uses RecurrentPPO for temporal memory across steps
    # A flag file tracks which model type was used so we load correctly
    curriculum_flag = session_path / "curriculum_enabled.flag"

    if checkpoint_path and resume:
        # Resume from existing checkpoint
        print(f"Resuming from checkpoint: {checkpoint_path} (step {start_steps:,})")

        # Detect model type from flag file
        # If the flag file exists, the checkpoint was saved by RecurrentPPO
        is_recurrent = curriculum_flag.exists()

        if is_recurrent:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(checkpoint_path, env=env)
        else:
            model = PPO.load(checkpoint_path, env=env)

        # Calculate remaining steps (may be 0 if already completed)
        remaining_steps = max(0, total_timesteps - start_steps)
    else:
        # Start fresh training with new model
        print("Starting fresh training...")

        if use_curriculum:
            # Use RecurrentPPO with LSTM for curriculum learning
            # LSTM provides temporal memory across longer episodes
            from sb3_contrib import RecurrentPPO

            # RecurrentPPO needs shorter rollouts for effective LSTM BPTT.
            # 128 steps causes gradient vanishing through the LSTM
            # (0.9^128 ≈ 1e-6); 32 is within effective LSTM memory range
            # (0.9^32 ≈ 0.035). Rollouts happen 4× more often to compensate.
            # Math: 8 envs × 32 steps = 256 samples/rollout,
            # 256/128 = 2 minibatches, 2 × 3 epochs = 6 gradient steps.
            recurrent_n_steps = 32
            recurrent_batch_size = 128

            model = RecurrentPPO(
                # MultiInputLstmPolicy: Dict obs + LSTM layers
                "MultiInputLstmPolicy",
                env,
                # Rollout parameters — shorter for LSTM gradient flow
                n_steps=recurrent_n_steps,
                batch_size=recurrent_batch_size,
                n_epochs=config.n_epochs,
                # Discount and advantage estimation
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                # PPO clipping
                clip_range=config.clip_range,
                # Loss coefficients
                ent_coef=config.ent_coef,
                vf_coef=config.vf_coef,
                # Optimizer
                learning_rate=config.learning_rate,
                # LSTM network architecture (shared LSTM)
                policy_kwargs=create_lstm_policy_kwargs(),
                verbose=1,
                seed=seed,
            )

            # Write flag file so we know to use RecurrentPPO.load() on resume
            curriculum_flag.write_text("curriculum")
        else:
            # Standard PPO without LSTM
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
    callback_list: list[BaseCallback] = [
        # Save model checkpoints periodically
        # Saves model_100000.zip, model_200000.zip, etc.
        CumulativeCheckpointCallback(
            save_path=session_path / "checkpoints",
            save_freq=100_000,  # Save every 100K steps
            vec_normalize=env,  # Save normalization stats with each checkpoint
        ),
        # Log game-specific metrics to TensorBoard
        # Tracks badges, events, exploration progress
        TensorboardCallback(log_freq=1000, session_path=session_path),
        # Warn if training appears stalled
        # Detects when reward stops improving
        StallDetectionCallback(check_freq=50_000),
        # Log training speed (steps/second)
        # Helps identify performance issues
        PerformanceCallback(log_freq=10_000),
    ]

    # Optionally add wandb callback for experiment tracking
    if wandb_run is not None:
        callback_list.append(WandbCallback(wandb_run=wandb_run))

    callbacks = CallbackList(callback_list)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print(f"Training for {remaining_steps:,} steps...")

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
        env.save(str(vecnorm_path))
        print(f"Saved final model: {final_path}")

        # Upload final model as wandb Artifact for easy access/download
        if wandb_run is not None:
            import wandb as _wandb

            artifact = _wandb.Artifact(
                name=f"model-{session_path.name}",  # type: ignore[attr-defined]
                type="model",
                description=f"Final KantoRL model at step {model.num_timesteps:,}",
            )
            artifact.add_file(str(final_path))
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
            print("Wandb run finished and artifact uploaded.")

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
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning with auto-checkpointing, HM automation, and LSTM",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Enable modular agent (quest planner, navigator, battler)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming to shared map visualization server",
    )
    parser.add_argument(
        "--stream-user",
        default="KantoRL",
        help="Username for stream display (default: KantoRL)",
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

    # -------------------------------------------------------------------------
    # Wandb Options
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases experiment tracking",
    )
    parser.add_argument(
        "--wandb-project",
        default="kantorl",
        help="wandb project name (default: kantorl)",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="wandb entity/team (default: personal account)",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="wandb run group for organizing experiments",
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
        use_curriculum=args.curriculum,
        use_agent=args.agent,
        enable_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
    )


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Allow running directly with: python -m kantorl.train
if __name__ == "__main__":
    main()
