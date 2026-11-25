"""Pre-registered checkpoint selection for the clean from-scratch run.

Selection rule (fixed before the run):
  1. Evaluate the 10M checkpoints on validation seeds 0-2 at 61,440 steps.
  2. Rank by Boulder reach, then Mt. Moon 3 Trainer 2 reach, then Mt. Moon
     B2F reach, then lower Boulder median.
  3. Evaluate the selected checkpoint on held-out seeds 10-19.

The final held-out report is the clean headline; the validation ranking only
picks the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def find_10m_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("model_*.zip"):
        try:
            steps = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if steps > 0 and steps % 10_000_000 == 0:
            checkpoints.append((steps, path))
    return sorted(checkpoints)


def evaluate(
    checkpoint: Path,
    out_dir: Path,
    seeds: list[int],
    budget: int,
    rom: str,
    save_state: str | None,
) -> None:
    if (out_dir / "milestones.json").exists():
        return
    cmd = [
        sys.executable,
        "-m",
        "kantorl",
        "milestones",
        str(checkpoint),
        rom,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--budget",
        str(budget),
        "--stochastic",
        "--out",
        str(out_dir),
    ]
    if save_state:
        cmd += ["--save-state", save_state]
    print(f"eval {checkpoint.name} -> {out_dir.name}", flush=True)
    subprocess.run(cmd, check=False)


def score(out_dir: Path) -> tuple[int, int, int, float]:
    rows = list(
        csv.DictReader(
            (out_dir / "milestones_summary.csv").open(newline="", encoding="utf-8")
        )
    )
    by_label = {row["label"]: row for row in rows}

    def reached(label: str) -> int:
        row = by_label.get(label)
        return int(row["reached_count"]) if row else 0

    def median(label: str) -> float:
        row = by_label.get(label)
        if not row or not row["median_first_step"]:
            return float("inf")
        return float(row["median_first_step"])

    boulder_median = median("Boulder Badge")
    # Higher is better; negate the median so lower medians rank higher.
    return (
        reached("Boulder Badge"),
        reached("Beat Mt. Moon 3 Trainer 2"),
        reached("Mt Moon B2F"),
        -boulder_median,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom_path")
    parser.add_argument("--run", required=True)
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--budget", type=int, default=61440)
    parser.add_argument(
        "--validation-seeds", nargs="+", type=int, default=[0, 1, 2]
    )
    parser.add_argument(
        "--final-seeds",
        nargs="+",
        type=int,
        default=list(range(10, 20)),
    )
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run)
    checkpoints = find_10m_checkpoints(run_dir)
    if not checkpoints:
        print(f"No 10M checkpoints in {run_dir}")
        return 1

    ranked: list[tuple[tuple[int, int, int, float], int, Path]] = []
    for steps, checkpoint in checkpoints:
        out_dir = run_dir / f"eval_clean_{steps}"
        evaluate(
            checkpoint,
            out_dir,
            args.validation_seeds,
            args.budget,
            args.rom_path,
            args.save_state,
        )
        if (out_dir / "milestones_summary.csv").exists():
            ranked.append((score(out_dir), steps, checkpoint))

    if not ranked:
        print("No validation reports found")
        return 1

    ranked.sort(key=lambda item: item[0], reverse=True)
    print("\nValidation ranking (higher is better):")
    for rank, (metrics, steps, _) in enumerate(ranked, start=1):
        print(
            f"  {rank:2d}. {steps / 1_000_000:6.1f}M  "
            f"Boulder={metrics[0]}, t2={metrics[1]}, B2F={metrics[2]}, "
            f"boulder_median={-metrics[3]:.0f}"
        )

    _, best_steps, best_checkpoint = ranked[0]
    print(f"\nSelected: {best_checkpoint.name} ({best_steps:,} steps)")
    if args.skip_final:
        return 0

    final_dir = run_dir / "eval_final_10x60k"
    evaluate(
        best_checkpoint,
        final_dir,
        args.final_seeds,
        args.budget,
        args.rom_path,
        args.save_state,
    )
    print(f"Held-out report: {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
