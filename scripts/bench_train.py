"""Find the fastest PPO configuration on this machine.

``bench_throughput.py`` measures the environments on their own. That number is
an upper bound: SB3 collects a rollout, then updates on the same CPU while the
workers sit idle. This script measures the two phases separately so the ratio is
visible, and sweeps the settings that actually move total throughput:

    - env count          (thread contention with the policy)
    - torch thread count (BLAS parallelism vs. stealing cores from workers)
    - n_steps            (how often the update runs)

Nothing here changes what the agent learns except ``n_steps`` and the env
count, which are both ordinary PPO hyperparameters.

Usage:
    python scripts/bench_train.py "C:/roms/Pokemon Red.gb"
    python scripts/bench_train.py ROM --envs 16 --threads 8 --n-steps 128
    python scripts/bench_train.py ROM --sweep
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run_config(
    rom_path: str,
    n_envs: int,
    rollout_threads: int,
    update_threads: int,
    n_steps: int,
    n_iters: int,
    seed: int = 0,
    with_harness: bool = False,
) -> dict:
    """Run ``n_iters`` full rollout+update cycles and time each phase.

    The two phases are sequential, so the thread count is retuned between them:
    a small batch of 16 frames is fastest on one or two threads, while the
    gradient update wants every core available. Setting one global thread count
    forces a compromise and costs throughput.

    With ``with_harness`` the environment and callbacks are wrapped exactly as
    ``train.py`` does (reward normalization plus the monitoring callbacks), so
    the difference against ``with_harness=False`` is the overhead of the real
    training path.
    """
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from kantorl.config import KantoConfig
    from kantorl.env import make_env
    from kantorl.train import create_policy_kwargs

    torch.set_num_threads(rollout_threads)

    config = KantoConfig(rom_path=rom_path, headless=True, n_envs=n_envs)
    env_fns = [
        make_env(rom_path, config, rank=i, seed=seed, reward_fn="default")
        for i in range(n_envs)
    ]
    vec = SubprocVecEnv(env_fns)

    try:
        if with_harness:
            from stable_baselines3.common.vec_env import VecNormalize

            vec = VecNormalize(
                vec, norm_obs=False, norm_reward=True, clip_reward=10.0
            )

        model = PPO(
            "MultiInputPolicy",
            vec,
            n_steps=n_steps,
            batch_size=512,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            learning_rate=config.learning_rate,
            policy_kwargs=create_policy_kwargs(),
            verbose=0,
            seed=seed,
            device="cpu",
        )

        model.learn(total_timesteps=n_envs * n_steps, progress_bar=False)  # warmup

        # ``collect_rollouts`` needs a callback; this one is a no-op.
        class _NoopCallback(BaseCallback):
            def _on_step(self) -> bool:
                return True

        if with_harness:
            from pathlib import Path as _Path

            from stable_baselines3.common.callbacks import CallbackList

            from kantorl.callbacks import (
                CumulativeCheckpointCallback,
                PerformanceCallback,
                StallDetectionCallback,
                TensorboardCallback,
            )

            session = _Path("runs/_bench_harness")
            callback = CallbackList(
                [
                    CumulativeCheckpointCallback(
                        save_path=session / "checkpoints", save_freq=100_000
                    ),
                    TensorboardCallback(log_freq=1000, session_path=session),
                    StallDetectionCallback(check_freq=50_000),
                    PerformanceCallback(log_freq=10_000),
                ]
            )
        else:
            callback = _NoopCallback()
        callback.init_callback(model)

        rollout_s = update_s = 0.0
        for _ in range(n_iters):
            torch.set_num_threads(rollout_threads)
            t0 = time.perf_counter()
            model.collect_rollouts(
                model.env,
                callback=callback,
                rollout_buffer=model.rollout_buffer,
                n_rollout_steps=model.n_steps,
            )
            t1 = time.perf_counter()
            torch.set_num_threads(update_threads)
            model.train()
            t2 = time.perf_counter()
            rollout_s += t1 - t0
            update_s += t2 - t1
    finally:
        vec.close()

    per_iter = n_envs * n_steps
    total_s = rollout_s + update_s
    return {
        "n_envs": n_envs,
        "rollout_threads": rollout_threads,
        "update_threads": update_threads,
        "n_steps": n_steps,
        "steps_per_s": n_iters * per_iter / total_s,
        "rollout_steps_per_s": n_iters * per_iter / rollout_s,
        "rollout_s": rollout_s / n_iters,
        "update_s": update_s / n_iters,
        "update_share": update_s / total_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom_path")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--rollout-threads", type=int, default=2)
    parser.add_argument("--update-threads", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--with-harness",
        action="store_true",
        help="Wrap env and callbacks exactly as train.py does",
    )
    parser.add_argument("--sweep", action="store_true", help="Sweep a grid")
    parser.add_argument(
        "--sweep-envs",
        default="8,12,16",
        help="Comma-separated env counts for --sweep",
    )
    parser.add_argument(
        "--sweep-rollout-threads",
        default="1,2,4",
        help="Comma-separated rollout thread counts for --sweep",
    )
    parser.add_argument(
        "--sweep-update-threads",
        default="8,16",
        help="Comma-separated update thread counts for --sweep",
    )
    parser.add_argument(
        "--sweep-nsteps",
        default="128,512",
        help="Comma-separated n_steps values for --sweep",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if not Path(args.rom_path).is_file():
        print(f"ROM not found: {args.rom_path}")
        return 1

    if args.sweep:
        envs = [int(v) for v in args.sweep_envs.split(",")]
        roll = [int(v) for v in args.sweep_rollout_threads.split(",")]
        upd = [int(v) for v in args.sweep_update_threads.split(",")]
        n_steps_list = [int(v) for v in args.sweep_nsteps.split(",")]
        grid = list(itertools.product(envs, roll, upd, n_steps_list))
    else:
        grid = [(args.envs, args.rollout_threads, args.update_threads, args.n_steps)]

    results = []
    header = (
        f"{'envs':>5} {'rthr':>5} {'uthr':>5} {'nstep':>6} {'harness':>8} "
        f"{'steps/s':>9} {'roll/s':>8} {'upd s':>7} {'upd%':>6}"
    )
    print(header)
    print("-" * len(header))

    for n_envs, rollout_threads, update_threads, n_steps in grid:
        try:
            row = run_config(
                args.rom_path,
                n_envs,
                rollout_threads,
                update_threads,
                n_steps,
                args.iters,
                with_harness=args.with_harness,
            )
        except Exception as exc:  # noqa: BLE001 - keep the sweep going
            print(
                f"{n_envs:>5} {rollout_threads:>5} {update_threads:>5} "
                f"{n_steps:>6}  failed: {exc}"
            )
            continue

        print(
            f"{row['n_envs']:>5} {row['rollout_threads']:>5} "
            f"{row['update_threads']:>5} {row['n_steps']:>6} "
            f"{'yes' if args.with_harness else 'no':>8} "
            f"{row['steps_per_s']:>9,.0f} {row['rollout_steps_per_s']:>8,.0f} "
            f"{row['update_s']:>7.2f} {row['update_share']*100:>5.0f}%"
        )
        results.append(row)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")

    if results:
        best = max(results, key=lambda r: r["steps_per_s"])
        print(
            f"\nFastest: {best['n_envs']} envs, "
            f"{best['rollout_threads']}/{best['update_threads']} rollout/update "
            f"threads, n_steps={best['n_steps']} -> "
            f"{best['steps_per_s']:,.0f} steps/s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
