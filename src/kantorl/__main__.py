"""
CLI entry point for KantoRL.

This module provides the main command-line interface for the KantoRL package.
It serves as the entry point when the package is invoked via:
- `kantorl <command>` (installed script)
- `python -m kantorl <command>` (module execution)

Architecture Role:
    This module is the user-facing interface to all KantoRL functionality.
    It parses command-line arguments and dispatches to the appropriate
    submodules (train.py, eval.py, benchmarks/). This follows the standard CLI
    pattern of having a thin entry point that delegates to implementation modules.

    User Input → __main__.py → train.py / eval.py / benchmarks → Core modules

Available Commands:
    train: Train a new agent or resume training from a checkpoint
    eval: Evaluate a trained agent and generate performance metrics
    info: Display environment information and configuration defaults
    benchmark: Run benchmark comparisons and hyperparameter search

Command Structure:
    kantorl <command> [arguments] [options]

    Examples:
        kantorl train pokemon_red.gb
        kantorl train pokemon_red.gb --envs 8 --steps 5000000
        kantorl eval checkpoint.zip pokemon_red.gb --render
        kantorl info pokemon_red.gb
        kantorl benchmark run config.yaml pokemon_red.gb
        kantorl benchmark search pokemon_red.gb --trials 20

Design Decisions:
    - Uses argparse subparsers for clean command separation
    - Lazy imports to minimize startup time (heavy imports only when needed)
    - Returns exit codes (0=success, 1=error) for shell scripting
    - Mirrors arguments from train.py/eval.py for consistency

Usage:
    # Train an agent (default: 16 envs, 10M steps)
    kantorl train path/to/pokemon_red.gb

    # Train with custom settings
    kantorl train path/to/pokemon_red.gb --envs 8 --steps 5000000 --session my_run

    # Resume from checkpoint (default behavior)
    kantorl train path/to/pokemon_red.gb

    # Start fresh training (ignore checkpoints)
    kantorl train path/to/pokemon_red.gb --no-resume

    # Train with streaming visualization
    kantorl train path/to/pokemon_red.gb --stream --stream-user "trainer1"

    # Evaluate a trained model
    kantorl eval runs/checkpoints/model_1000000.zip pokemon_red.gb

    # Evaluate with rendering (watch the agent play)
    kantorl eval checkpoint.zip pokemon_red.gb --render

    # Show environment info
    kantorl info
    kantorl info pokemon_red.gb  # Include observation/action space details

    # Alternative invocation via Python module
    python -m kantorl train path/to/pokemon_red.gb

Dependencies:
    - argparse: Command-line argument parsing
    - sys: Exit code handling
    - kantorl.train: Training functionality (lazy import)
    - kantorl.eval: Evaluation functionality (lazy import)
    - kantorl.config: Configuration display (lazy import)
    - kantorl.env: Environment info display (lazy import)
"""

import argparse
import sys

# =============================================================================
# MAIN CLI FUNCTION
# =============================================================================


def main() -> int:
    """
    Main CLI entry point for KantoRL.

    Parses command-line arguments using argparse subparsers and dispatches
    to the appropriate handler function. Uses lazy imports to minimize
    startup time - heavy modules (PyTorch, stable-baselines3) are only
    imported when their respective commands are invoked.

    Returns:
        Exit code: 0 for success, 1 for errors or unknown commands.
        This allows shell scripts to check command success via $?.

    Command Handlers:
        train: Calls kantorl.train.train() with parsed arguments
        eval: Calls kantorl.eval.evaluate() with parsed arguments
        info: Displays version and configuration information

    Example:
        >>> # This is typically called via the console script
        >>> sys.exit(main())
    """
    # -------------------------------------------------------------------------
    # Create Main Parser
    # -------------------------------------------------------------------------
    # The main parser handles the top-level program and subcommand dispatch
    parser = argparse.ArgumentParser(
        prog="kantorl",
        description="KantoRL - The MNIST of Pokemon Red for Reinforcement Learning",
    )

    # Create subparser container for commands
    # dest="command" stores which subcommand was chosen in args.command
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------------------------------------------------
    # Train Command
    # -------------------------------------------------------------------------
    # Train a PPO agent on Pokemon Red
    train_parser = subparsers.add_parser(
        "train",
        help="Train an agent on Pokemon Red",
    )

    # Required argument: ROM path
    train_parser.add_argument(
        "rom_path",
        help="Path to Pokemon Red ROM file (.gb)",
    )

    # Training configuration options
    train_parser.add_argument(
        "--session", "-s",
        default="runs",
        help="Session directory for checkpoints/logs (default: runs)",
    )
    train_parser.add_argument(
        "--steps", "-n",
        type=int,
        default=10_000_000,
        help="Total training steps (default: 10,000,000)",
    )
    train_parser.add_argument(
        "--envs", "-e",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16)",
    )
    train_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't resume from checkpoint",
    )
    train_parser.add_argument(
        "--reward", "-r",
        default="default",
        choices=["default", "badges_only", "exploration"],
        help="Reward function to use (default: default)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Curriculum learning option
    train_parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning with auto-checkpointing, HM automation, and LSTM",
    )

    # Streaming visualization options
    # These enable real-time visualization of training progress
    train_parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming to shared map visualization",
    )
    train_parser.add_argument(
        "--stream-user",
        default="KantoRL",
        help="Username for stream display (default: KantoRL)",
    )
    train_parser.add_argument(
        "--stream-color",
        default="#ff0000",
        help="Hex color for stream display (default: #ff0000)",
    )
    train_parser.add_argument(
        "--stream-sprite",
        type=int,
        default=0,
        help="Sprite ID (0-50) for stream display (default: 0)",
    )

    # -------------------------------------------------------------------------
    # Eval Command
    # -------------------------------------------------------------------------
    # Evaluate a trained agent
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a trained agent",
    )

    # Required arguments
    eval_parser.add_argument(
        "checkpoint",
        help="Path to model checkpoint (.zip)",
    )
    eval_parser.add_argument(
        "rom_path",
        help="Path to Pokemon Red ROM file (.gb)",
    )

    # Evaluation options
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    eval_parser.add_argument(
        "--render",
        action="store_true",
        help="Render gameplay (opens visualization window)",
    )

    # -------------------------------------------------------------------------
    # Info Command
    # -------------------------------------------------------------------------
    # Display environment information
    info_parser = subparsers.add_parser(
        "info",
        help="Show environment information",
    )

    # Optional ROM path for detailed space information
    info_parser.add_argument(
        "rom_path",
        nargs="?",  # Optional positional argument
        help="Path to ROM (optional, enables observation/action space display)",
    )

    # -------------------------------------------------------------------------
    # Benchmark Command
    # -------------------------------------------------------------------------
    # Benchmark system for comparing RL configurations
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarks to compare RL configurations",
    )

    # Create benchmark subcommands
    benchmark_subparsers = benchmark_parser.add_subparsers(
        dest="benchmark_command",
        help="Benchmark operations",
    )

    # -------------------------------------------------------------------------
    # Benchmark Run Command
    # -------------------------------------------------------------------------
    benchmark_run_parser = benchmark_subparsers.add_parser(
        "run",
        help="Run batch benchmark comparison from config file",
    )
    benchmark_run_parser.add_argument(
        "config_path",
        help="Path to benchmark config YAML file",
    )
    benchmark_run_parser.add_argument(
        "rom_path",
        help="Path to Pokemon Red ROM file (.gb)",
    )
    benchmark_run_parser.add_argument(
        "--output", "-o",
        default="runs/benchmarks",
        help="Output directory for benchmark results (default: runs/benchmarks)",
    )
    benchmark_run_parser.add_argument(
        "--tier", "-t",
        default="bronze",
        choices=["bronze", "silver", "gold", "champion"],
        help="Milestone tier to benchmark against (default: bronze)",
    )
    benchmark_run_parser.add_argument(
        "--max-steps", "-n",
        type=int,
        default=2_000_000,
        help="Maximum training steps per run (default: 2,000,000)",
    )

    # -------------------------------------------------------------------------
    # Benchmark Single Command
    # -------------------------------------------------------------------------
    benchmark_single_parser = benchmark_subparsers.add_parser(
        "single",
        help="Run a single benchmark training run",
    )
    benchmark_single_parser.add_argument(
        "rom_path",
        help="Path to Pokemon Red ROM file (.gb)",
    )
    benchmark_single_parser.add_argument(
        "--tier", "-t",
        default="bronze",
        choices=["bronze", "silver", "gold", "champion"],
        help="Milestone tier to benchmark against (default: bronze)",
    )
    benchmark_single_parser.add_argument(
        "--max-steps", "-n",
        type=int,
        default=2_000_000,
        help="Maximum training steps (default: 2,000,000)",
    )
    benchmark_single_parser.add_argument(
        "--envs", "-e",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16)",
    )
    benchmark_single_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    benchmark_single_parser.add_argument(
        "--output", "-o",
        default="runs/benchmarks",
        help="Output directory for benchmark results (default: runs/benchmarks)",
    )
    benchmark_single_parser.add_argument(
        "--save-state",
        default=None,
        help="Path to save state file (.state) for starting point",
    )

    # Streaming visualization options (same as train command)
    benchmark_single_parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming to shared map visualization",
    )
    benchmark_single_parser.add_argument(
        "--stream-user",
        default="KantoRL-bench",
        help="Username for stream display (default: KantoRL-bench)",
    )
    benchmark_single_parser.add_argument(
        "--stream-color",
        default="#ff0000",
        help="Hex color for stream display (default: #ff0000)",
    )
    benchmark_single_parser.add_argument(
        "--stream-sprite",
        type=int,
        default=0,
        help="Sprite ID (0-50) for stream display (default: 0)",
    )

    # -------------------------------------------------------------------------
    # Benchmark Compare Command
    # -------------------------------------------------------------------------
    benchmark_compare_parser = benchmark_subparsers.add_parser(
        "compare",
        help="Compare past benchmark results from JSON files",
    )
    benchmark_compare_parser.add_argument(
        "results_dir",
        help="Directory containing benchmark result JSON files",
    )

    # -------------------------------------------------------------------------
    # Benchmark Search Command (Optuna)
    # -------------------------------------------------------------------------
    benchmark_search_parser = benchmark_subparsers.add_parser(
        "search",
        help="Run Optuna hyperparameter search (requires: pip install optuna)",
    )
    benchmark_search_parser.add_argument(
        "rom_path",
        help="Path to Pokemon Red ROM file (.gb)",
    )
    benchmark_search_parser.add_argument(
        "--trials", "-t",
        type=int,
        default=50,
        help="Number of Optuna trials to run (default: 50)",
    )
    benchmark_search_parser.add_argument(
        "--tier",
        default="bronze",
        choices=["bronze", "silver", "gold", "champion"],
        help="Milestone tier to optimize for (default: bronze)",
    )
    benchmark_search_parser.add_argument(
        "--max-steps", "-n",
        type=int,
        default=2_000_000,
        help="Maximum training steps per trial (default: 2,000,000)",
    )
    benchmark_search_parser.add_argument(
        "--envs", "-e",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16)",
    )
    benchmark_search_parser.add_argument(
        "--output", "-o",
        default="runs/optuna",
        help="Output directory for search results (default: runs/optuna)",
    )
    benchmark_search_parser.add_argument(
        "--study-name",
        default="kantorl_benchmark",
        help="Optuna study name (default: kantorl_benchmark)",
    )
    benchmark_search_parser.add_argument(
        "--save-state",
        default=None,
        help="Path to save state file (.state) for starting point",
    )

    # -------------------------------------------------------------------------
    # Parse Arguments
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Command Dispatch
    # -------------------------------------------------------------------------
    # Each command has its own handler with lazy imports to minimize startup time

    if args.command == "train":
        # ---------------------------------------------------------------------
        # Train Command Handler
        # ---------------------------------------------------------------------
        # Lazy import: train module pulls in PyTorch, stable-baselines3, etc.
        # Only imported when train command is actually used
        from kantorl.train import train

        # Call training function with parsed arguments
        train(
            rom_path=args.rom_path,
            session_path=args.session,
            total_timesteps=args.steps,
            n_envs=args.envs,
            resume=not args.no_resume,  # --no-resume inverts the default True
            reward_fn=args.reward,
            seed=args.seed,
            enable_streaming=args.stream,
            stream_username=args.stream_user,
            stream_color=args.stream_color,
            stream_sprite_id=args.stream_sprite,
            use_curriculum=args.curriculum,
        )
        return 0  # Success

    elif args.command == "eval":
        # ---------------------------------------------------------------------
        # Eval Command Handler
        # ---------------------------------------------------------------------
        # Lazy import: eval module also pulls in heavy dependencies
        from kantorl.eval import evaluate

        # Call evaluation function with parsed arguments
        evaluate(
            checkpoint_path=args.checkpoint,
            rom_path=args.rom_path,
            n_episodes=args.episodes,
            render=args.render,
        )
        return 0  # Success

    elif args.command == "info":
        # ---------------------------------------------------------------------
        # Info Command Handler
        # ---------------------------------------------------------------------
        # Display version and configuration information
        # Useful for debugging and understanding the environment
        from kantorl import __version__
        from kantorl.config import KantoConfig
        from kantorl.env import KantoRedEnv

        # Print version banner
        print(f"KantoRL v{__version__}")
        print()

        # Print default configuration values
        # These are the key settings users might want to know
        print("Default Configuration:")
        config = KantoConfig()
        for field_name in ["action_freq", "max_steps", "frame_stacks", "screen_size"]:
            print(f"  {field_name}: {getattr(config, field_name)}")
        print()

        # If ROM path provided, show observation and action space details
        # This requires creating an environment instance
        if args.rom_path:
            # Create environment to inspect its spaces
            env = KantoRedEnv(rom_path=args.rom_path)

            # Print observation space breakdown
            # The observation is a Dict space with multiple components
            print("Observation Space:")
            for key, space in env.observation_space.spaces.items():
                print(f"  {key}: {space}")
            print()

            # Print action space
            print(f"Action Space: {env.action_space}")

            # Clean up environment
            env.close()

        return 0  # Success

    elif args.command == "benchmark":
        # ---------------------------------------------------------------------
        # Benchmark Command Handler
        # ---------------------------------------------------------------------
        # Handle benchmark subcommands

        if args.benchmark_command == "run":
            # -----------------------------------------------------------------
            # Benchmark Run: Execute batch comparison from config file
            # -----------------------------------------------------------------
            from kantorl.benchmarks.reporters import report_all
            from kantorl.benchmarks.runner import run_from_config
            from kantorl.benchmarks.scenarios import MilestoneTier

            tier = MilestoneTier(args.tier)

            # Run benchmarks from config file
            results = run_from_config(
                config_path=args.config_path,
                rom_path=args.rom_path,
                output_dir=args.output,
                verbose=1,
            )

            # Generate all report formats
            report_all(
                results=results,
                output_dir=args.output,
                tier=args.tier,
                max_steps=args.max_steps,
            )

            return 0  # Success

        elif args.benchmark_command == "single":
            # -----------------------------------------------------------------
            # Benchmark Single: Run a single benchmark training run
            # -----------------------------------------------------------------
            from kantorl.benchmarks.reporters import ConsoleReporter
            from kantorl.benchmarks.runner import BenchmarkRunner
            from kantorl.benchmarks.scenarios import MilestoneTier

            tier = MilestoneTier(args.tier)

            # Create runner and execute
            runner = BenchmarkRunner(
                rom_path=args.rom_path,
                tier=tier,
                max_steps=args.max_steps,
                n_envs=args.envs,
                output_dir=args.output,
                save_state_path=args.save_state,
                enable_streaming=args.stream,
                stream_username=args.stream_user,
                stream_color=args.stream_color,
                stream_sprite_id=args.stream_sprite,
                verbose=1,
            )

            result = runner.run_single(
                config_name="single_run",
                seed=args.seed,
            )

            # Print summary
            print(f"\n{'='*60}")
            print("BENCHMARK COMPLETE")
            print(f"{'='*60}")
            print(f"  Final badges: {result.final_badges}")
            print(f"  Final events: {result.final_events}")
            print(f"  Total steps: {result.total_steps:,}")
            print(f"  Steps to badge 1: {result.first_badge_steps or 'N/A'}")

            return 0  # Success

        elif args.benchmark_command == "compare":
            # -----------------------------------------------------------------
            # Benchmark Compare: Compare past benchmark results
            # -----------------------------------------------------------------
            import json
            from pathlib import Path

            from kantorl.benchmarks.metrics import BenchmarkResult
            from kantorl.benchmarks.reporters import ConsoleReporter

            results_dir = Path(args.results_dir)

            # Load all JSON result files
            all_results: dict[str, list[BenchmarkResult]] = {}
            for json_file in results_dir.glob("*.json"):
                with open(json_file) as f:
                    data = json.load(f)

                # Extract results from JSON structure
                for config_name, config_results in data.get("results", {}).items():
                    if config_name not in all_results:
                        all_results[config_name] = []
                    for result_dict in config_results:
                        all_results[config_name].append(
                            BenchmarkResult.from_dict(result_dict)
                        )

            if not all_results:
                print(f"No benchmark results found in: {results_dir}")
                return 1

            # Report comparison
            reporter = ConsoleReporter()
            reporter.report_results(all_results)

            return 0  # Success

        elif args.benchmark_command == "search":
            # -----------------------------------------------------------------
            # Benchmark Search: Optuna hyperparameter optimization
            # -----------------------------------------------------------------
            from kantorl.benchmarks.optuna_search import run_search
            from kantorl.benchmarks.scenarios import MilestoneTier

            tier = MilestoneTier(args.tier)

            # Run Optuna search (results are printed by run_search)
            run_search(
                rom_path=args.rom_path,
                n_trials=args.trials,
                tier=tier,
                max_steps=args.max_steps,
                n_envs=args.envs,
                output_dir=args.output,
                study_name=args.study_name,
                save_state_path=args.save_state,
                verbose=1,
            )

            return 0  # Success

        else:
            # No benchmark subcommand specified
            benchmark_parser.print_help()
            return 1

    else:
        # ---------------------------------------------------------------------
        # No Command / Unknown Command
        # ---------------------------------------------------------------------
        # Print help message if no command specified
        parser.print_help()
        return 1  # Error exit code


# =============================================================================
# MODULE EXECUTION
# =============================================================================

# Allow execution via: python -m kantorl
# This calls main() and exits with the returned code
if __name__ == "__main__":
    sys.exit(main())
