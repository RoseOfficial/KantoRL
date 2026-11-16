"""Measure a uniform-random policy as the control for milestone evaluations.

A trained policy is only interesting relative to chance. This runs the same
episodes a ``kantorl milestones`` evaluation would, with the policy replaced by
uniform action sampling, and writes the same report shape so the two can be
compared directly with ``scripts/compare_milestones.py``.

Usage:
    python scripts/random_baseline.py "C:/roms/Pokemon Red.gb" --seeds 0 1 2
    python scripts/random_baseline.py ROM --seeds 0 1 2 3 4 --budget 20000

Each seed is one episode, matching the evaluation harness: same starting state,
same step budget. Reward is the default reward; it does not influence behaviour
here, only the reported number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kantorl.config import KantoConfig  # noqa: E402
from kantorl.env import KantoRedEnv  # noqa: E402
from kantorl.milestones import (  # noqa: E402
    CORE_MILESTONE_KEYS,
    EpisodeResult,
    MilestoneReport,
    MilestoneTracker,
    _milestone_from_key,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom_path")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--out", default="runs/baseline_random")
    parser.add_argument("--obs", nargs="+", default=None)
    args = parser.parse_args()

    config = KantoConfig(rom_path=args.rom_path, max_steps=int(args.budget))
    if args.obs:
        config.obs_components = tuple(args.obs)
        config.validate()

    env = KantoRedEnv(config=config, reward_fn="default")
    episodes: list[EpisodeResult] = []
    try:
        for episode_index, seed in enumerate(args.seeds):
            obs, info = env.reset(seed=int(seed))
            tracker = MilestoneTracker()
            tracker.update(info)
            rng = np.random.default_rng(seed)

            total_reward = 0.0
            steps = 0
            while steps < args.budget:
                obs, reward, _terminated, truncated, info = env.step(
                    int(rng.integers(0, 8))
                )
                total_reward += float(reward)
                steps += 1
                tracker.update(info)
                if truncated:
                    break

            episodes.append(
                EpisodeResult(
                    episode=episode_index + 1,
                    seed=int(seed),
                    steps=steps,
                    reward=total_reward,
                    badges=int(info.get("badges", 0)),
                    events=int(info.get("events", 0)),
                    explore_tiles=int(info.get("explore_tiles", 0)),
                    party_count=int(info.get("party_count", 0)),
                    first_steps=tracker.first_steps,
                )
            )
            print(
                f"seed {seed}: {steps} steps, reward {total_reward:.1f}, "
                f"party {info.get('party_count', 0)}, "
                f"milestones {len(tracker.first_steps)}"
            )

        catalog = {
            key: _milestone_from_key(key) for key in CORE_MILESTONE_KEYS
        }
        for episode in episodes:
            for key in episode.first_steps:
                catalog.setdefault(key, _milestone_from_key(key))

        report = MilestoneReport(
            checkpoint="uniform-random",
            obs_components=tuple(config.obs_components),
            budget=int(args.budget),
            deterministic=False,
            episodes=episodes,
            catalog=catalog,
        )
    finally:
        env.close()

    write_report(report, args.out)
    print()
    print(report.format_summary())
    print(f"\nWrote {args.out}/milestones.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
