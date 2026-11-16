"""Measure throughput and memory for N parallel Pokemon Red environments.

Answers "how many envs can this machine run" with numbers instead of guesses.
The loop mirrors training exactly: ``SubprocVecEnv`` + the same ``make_env``
factory, so inter-process observation pickling is included in the measurement.

Usage:
    python scripts/bench_throughput.py "C:/Users/neoga/roms/Pokemon Red.gb"
    python scripts/bench_throughput.py ROM --envs 1 2 4 8 --steps 400
    python scripts/bench_throughput.py ROM --envs 8 --with-ppo
    python scripts/bench_throughput.py ROM --envs 16 --with-ppo --device cpu

Notes:
    - Uses the full seven-component observation by default. Leaner tiers send
      less data over IPC, so T0/T1 will be somewhat faster.
    - ``--with-ppo`` additionally measures the training loop, which is where a
      CPU-only torch install becomes the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from kantorl.config import KantoConfig  # noqa: E402
from kantorl.env import make_env  # noqa: E402


def process_tree_memory_mb() -> float:
    """Resident memory of this process plus all children, in MB."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is optional
        return float("nan")

    me = psutil.Process(os.getpid())
    total = me.memory_info().rss
    for child in me.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total / (1024**2)


def bench_envs(rom_path: str, n_envs: int, steps: int, warmup: int, seed: int = 0) -> dict:
    """Step ``n_envs`` environments in parallel and report throughput."""
    config = KantoConfig(rom_path=rom_path, headless=True)
    env_fns = [
        make_env(
            rom_path,
            config,
            rank=i,
            seed=seed,
            reward_fn="default",
            enable_streaming=False,
            enable_curriculum=False,
            enable_agent=False,
            initial_step_offset=0,
        )
        for i in range(n_envs)
    ]

    setup_start = time.perf_counter()
    vec = SubprocVecEnv(env_fns)
    vec.reset()
    setup_s = time.perf_counter() - setup_start
    mem_mb = process_tree_memory_mb()

    rng = np.random.default_rng(seed)
    actions = [rng.integers(0, 8, size=n_envs) for _ in range(64)]

    try:
        for i in range(warmup):
            vec.step(actions[i % len(actions)])

        start = time.perf_counter()
        for i in range(steps):
            vec.step(actions[i % len(actions)])
        elapsed = time.perf_counter() - start
    finally:
        vec.close()

    total_steps = steps * n_envs
    return {
        "n_envs": n_envs,
        "steps_per_env": steps,
        "wall_s": elapsed,
        "total_steps_per_s": total_steps / elapsed,
        "per_env_steps_per_s": (total_steps / elapsed) / n_envs,
        "setup_s": setup_s,
        "rss_mb": mem_mb,
    }


def bench_ppo(
    rom_path: str, n_envs: int, timesteps: int, seed: int = 0, device: str = "auto"
) -> dict:
    """Measure the full training loop (rollout + PPO update) for ``timesteps``."""
    from stable_baselines3 import PPO

    from kantorl.train import create_policy_kwargs

    config = KantoConfig(rom_path=rom_path, headless=True, n_envs=n_envs)
    env_fns = [
        make_env(rom_path, config, rank=i, seed=seed, reward_fn="default")
        for i in range(n_envs)
    ]

    vec = SubprocVecEnv(env_fns)
    try:
        model = PPO(
            "MultiInputPolicy",
            vec,
            n_steps=128,
            batch_size=512,
            n_epochs=3,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            learning_rate=config.learning_rate,
            policy_kwargs=create_policy_kwargs(),
            verbose=0,
            seed=seed,
            device=device,
        )
        device = str(model.device)

        start = time.perf_counter()
        model.learn(total_timesteps=timesteps, progress_bar=False)
        elapsed = time.perf_counter() - start
    finally:
        vec.close()

    return {
        "n_envs": n_envs,
        "timesteps": timesteps,
        "wall_s": elapsed,
        "train_steps_per_s": timesteps / elapsed,
        "device": device,
        "rss_mb": process_tree_memory_mb(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom_path")
    parser.add_argument(
        "--envs",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 12, 16],
        help="Environment counts to test (default: 1 2 4 6 8 12 16)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Measured steps per env (default: 300)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Unmeasured warmup steps per env (default: 100)",
    )
    parser.add_argument(
        "--with-ppo",
        action="store_true",
        help="Also measure the full PPO training loop",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Write raw results to this path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the PPO policy: auto, cpu or cuda (default: auto)",
    )
    parser.add_argument(
        "--ppo-envs",
        type=int,
        default=None,
        help="Env count for the PPO loop (default: the fastest row from the sweep)",
    )
    args = parser.parse_args()

    rom = Path(args.rom_path)
    if not rom.is_file():
        print(f"ROM not found: {rom}")
        return 1

    print(f"ROM: {rom}  ({rom.stat().st_size:,} bytes)")
    print(f"CPU: {os.cpu_count()} logical cores")
    print()

    results = []
    header = f"{'envs':>5} {'steps/s':>9} {'per-env':>8} {'scaling':>8} {'setup':>7} {'RSS':>8}"
    print(header)
    print("-" * len(header))

    baseline = None
    for n_envs in args.envs:
        try:
            row = bench_envs(rom.as_posix(), n_envs, args.steps, args.warmup)
        except Exception as exc:  # noqa: BLE001 - a failed config should not stop the sweep
            print(f"{n_envs:>5}  failed: {exc}")
            continue

        if baseline is None:
            baseline = row["total_steps_per_s"]
        scaling = row["total_steps_per_s"] / baseline
        print(
            f"{n_envs:>5} {row['total_steps_per_s']:>9,.0f} "
            f"{row['per_env_steps_per_s']:>8,.0f} "
            f"{scaling:>7.2f}x "
            f"{row['setup_s']:>6.1f}s "
            f"{row['rss_mb']:>7,.0f}M"
        )
        results.append(row)

    if args.with_ppo:
        print()
        best = max(results, key=lambda r: r["total_steps_per_s"]) if results else None
        n_envs = args.ppo_envs or (best["n_envs"] if best else 8)
        timesteps = max(n_envs * 128 * 3, 2048)
        print(
            f"PPO training loop: {n_envs} envs, {timesteps:,} timesteps, "
            f"device={args.device}"
        )
        try:
            ppo = bench_ppo(rom.as_posix(), n_envs, timesteps, device=args.device)
            print(
                f"  device      : {ppo['device']}\n"
                f"  wall        : {ppo['wall_s']:.1f}s\n"
                f"  train steps/s: {ppo['train_steps_per_s']:,.0f}\n"
                f"  RSS         : {ppo['rss_mb']:,.0f} MB"
            )
            results.append({"ppo": ppo})
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
