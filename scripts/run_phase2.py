"""Execute the pre-registered Phase 2 four-arm PPO ablation.

The design and decision rules are frozen in
``docs/PREREGISTRATION_PHASE2.md``. This script only runs them: four 20M
training arms (control, archive, tiles, both), then milestone evaluation of
the fixed 10M and 20M checkpoints on validation seeds 0-2.

Usage:
    python scripts/run_phase2.py "C:/roms/Pokemon Red.gb" \
        --save-state "C:/roms/states/bulba_start.state"
    python scripts/run_phase2.py ROM --save-state STATE --dry-run
    python scripts/run_phase2.py ROM --save-state STATE --skip-training
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS = (
    "phase2_control",
    "phase2_archive50",
    "phase2_tiles",
    "phase2_tiles_archive50",
)

CHECKPOINT_STEPS = (10_000_000, 20_000_000)


def session_name(tier: str, steps: int, tag: str, seed: int) -> str:
    """Match scripts/run_tiers.py's session naming exactly."""
    return f"{tier}_{steps // 1_000_000}M{tag}_s{seed}"


def run_training(
    rom: str,
    save_state: str,
    steps: int,
    seed: int,
    tag: str,
    arms: tuple[str, ...],
    governor_hours: float,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_tiers.py"),
        rom,
        "--tiers",
        *arms,
        "--seeds",
        str(seed),
        "--steps",
        str(steps),
        "--tag",
        tag,
        "--save-state",
        save_state,
        "--no-stream",
        "--governor-hours",
        f"{governor_hours:g}",
    ]
    if dry_run:
        cmd.append("--dry-run")

    print("\n== training ==", flush=True)
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def evaluate(
    rom: str,
    save_state: str,
    checkpoint: Path,
    out_dir: Path,
    seeds: list[int],
    budget: int,
    dry_run: bool,
) -> int:
    if (out_dir / "milestones.json").exists():
        print(f"skip eval {out_dir.name} (report exists)")
        return 0

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
        "--save-state",
        save_state,
    ]
    print(f"eval {out_dir.name} <- {checkpoint.name}", flush=True)
    if dry_run:
        print(" ".join(cmd), flush=True)
        return 0
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def first_badge_step(run_dir: Path) -> int | None:
    path = run_dir / "first_seen.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("kind") == "badge" and row.get("index") == "1":
                try:
                    return int(row["step"])
                except (KeyError, ValueError):
                    return None
    return None


def summarise(
    arms: tuple[str, ...],
    steps: int,
    tag: str,
    seed: int,
    eval_root: Path,
) -> int:
    print("\n== phase 2 summary ==")
    header = (
        f"{'arm':<26} {'ckpt':>5} {'boulder':>8} {'forest':>7} "
        f"{'pewter':>7} {'badges':>7} {'events':>7} {'train 1st':>10}"
    )
    print(header)
    print("-" * len(header))

    found = 0
    for arm in arms:
        run_dir = ROOT / "runs" / session_name(arm, steps, tag, seed)
        train_step = first_badge_step(run_dir)
        for checkpoint_step in CHECKPOINT_STEPS:
            out_dir = eval_root / f"{run_dir.name}_model_{checkpoint_step}"
            report_path = out_dir / "milestones.json"
            if not report_path.exists():
                print(
                    f"{arm:<26} {checkpoint_step // 1_000_000:>4}M "
                    f"{'missing':>8} {'':>7} {'':>7} {'':>7} {'':>7} "
                    f"{(train_step if train_step is not None else '-'):>10}"
                )
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            labels = {key: meta["label"] for key, meta in report["catalog"].items()}
            counts: dict[str, int] = {}
            for episode in report["episodes"]:
                for key in episode["first_steps"]:
                    label = labels.get(key, key)
                    counts[label] = counts.get(label, 0) + 1
            episodes = report["episodes"]
            n = len(episodes)
            boulder = counts.get("Boulder Badge", 0)
            forest = counts.get("Viridian Forest", 0)
            pewter = counts.get("Pewter City", 0)
            badges = sum(ep["badges"] for ep in episodes) / n if n else 0.0
            events = sum(ep["events"] for ep in episodes) / n if n else 0.0
            print(
                f"{arm:<26} {checkpoint_step // 1_000_000:>4}M "
                f"{boulder:>4}/{n:<3} {forest:>3}/{n:<3} {pewter:>3}/{n:<3} "
                f"{badges:>7.2f} {events:>7.1f} "
                f"{(train_step if train_step is not None else '-'):>10}"
            )
            found += 1

    if not found:
        print("\nNo evaluation reports found yet.")
        return 1

    print(
        "\nDecision rule: an arm at 2/3 or 3/3 Boulder indicates that "
        "component; Pewter 2/3+ without Boulder is a navigation gain."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom_path")
    parser.add_argument(
        "--save-state",
        required=True,
        help="Training start state; evaluation must use the same state.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=61_440)
    parser.add_argument("--tag", default="_phase2")
    parser.add_argument("--governor-hours", type=float, default=24.0)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=list(ARMS),
        help="Subset of the pre-registered arms (default: all four).",
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not Path(args.rom_path).is_file():
        print(f"ROM not found: {args.rom_path}")
        return 1
    if not Path(args.save_state).is_file():
        print(f"Save state not found: {args.save_state}")
        return 1

    arms = tuple(args.arms)
    eval_root = ROOT / "runs" / "phase2_eval"

    if not args.skip_training:
        code = run_training(
            args.rom_path,
            args.save_state,
            args.steps,
            args.seed,
            args.tag,
            arms,
            args.governor_hours,
            args.dry_run,
        )
        if code != 0:
            print(f"training sweep failed (exit {code})")
            return code

    if not args.skip_eval:
        print("\n== evaluation ==", flush=True)
        for arm in arms:
            run_dir = ROOT / "runs" / session_name(arm, args.steps, args.tag, args.seed)
            for checkpoint_step in CHECKPOINT_STEPS:
                checkpoint = run_dir / "checkpoints" / f"model_{checkpoint_step}.zip"
                if not checkpoint.exists():
                    print(
                        f"skip eval {run_dir.name} @ {checkpoint_step:,} "
                        "(no checkpoint)"
                    )
                    continue
                out_dir = eval_root / f"{run_dir.name}_model_{checkpoint_step}"
                code = evaluate(
                    args.rom_path,
                    args.save_state,
                    checkpoint,
                    out_dir,
                    args.eval_seeds,
                    args.budget,
                    args.dry_run,
                )
                if code != 0:
                    print(f"evaluation failed for {out_dir.name} (exit {code})")
                    return code

    if args.dry_run:
        print("\ndry run: commands printed, nothing executed")
        return 0

    return summarise(arms, args.steps, args.tag, args.seed, eval_root)


if __name__ == "__main__":
    raise SystemExit(main())
