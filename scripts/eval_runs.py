"""Evaluate a set of training runs and summarise each tier across seeds.

Replication means several training runs per tier, each evaluated on several
episode seeds. Reading nine milestone.json files by hand is how you end up
quoting one lucky run, so this does both steps and reports the spread.

For each run directory it:
  1. finds the latest checkpoint,
  2. runs the milestone harness into <run>/eval (unless already present),
  3. groups runs by tier and reports mean / min / max per milestone.

Usage:
    python scripts/eval_runs.py "C:/roms/Pokemon Red.gb" --runs "runs/t0_pure_20M_s*"
    python scripts/eval_runs.py ROM --runs "runs/*_20M_s*" --seeds 0 1 2 3 4 \
        --save-state "C:/roms/states/bulba_start.state"
    python scripts/eval_runs.py ROM --runs "runs/*_20M_s*" --report-only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

KEY_MILESTONES = (
    "Route 1",
    "Viridian City",
    "Route 2",
    "Viridian Forest",
    "Pewter City",
    "Route 22",
)


def latest_checkpoint(run: Path) -> Path | None:
    best, best_steps = None, -1
    for path in (run / "checkpoints").glob("model_*.zip"):
        try:
            steps = int(path.stem.removeprefix("model_"))
        except ValueError:
            continue
        if steps > best_steps:
            best, best_steps = path, steps
    return best


def best_checkpoint(run: Path) -> Path | None:
    """The model snapshot saved when the run last reached a new milestone."""
    path = run / "checkpoints" / "best_model.zip"
    return path if path.exists() else None


def tier_of(run: Path) -> str:
    return re.sub(r"_s\d+$", "", run.name)


def reach_counts(report: dict) -> dict[str, int]:
    cat = report["catalog"]
    label_of = {key: meta["label"] for key, meta in cat.items()}
    counts: dict[str, int] = {}
    for episode in report["episodes"]:
        for key in episode["first_steps"]:
            counts[label_of[key]] = counts.get(label_of[key], 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom_path")
    parser.add_argument("--runs", nargs="+", required=True, help="Glob(s) of run dirs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--save-state", default=None)
    parser.add_argument(
        "--map-player-marker",
        action="store_true",
        help="Enable the player-position marker (must match training)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Evaluate checkpoints/best_model.zip instead of the latest checkpoint",
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    runs: list[Path] = []
    for pattern in args.runs:
        runs.extend(sorted(p for p in Path().glob(pattern) if p.is_dir()))
    if not runs:
        print("No run directories matched.")
        return 1

    if not args.report_only:
        for run in runs:
            eval_dir = run / ("eval_best" if args.best else "eval")
            out = eval_dir / "milestones.json"
            if args.skip_existing and out.exists():
                print(f"skip {run.name} (eval exists)")
                continue
            checkpoint = best_checkpoint(run) if args.best else latest_checkpoint(run)
            if checkpoint is None:
                kind = "best_model" if args.best else "checkpoint"
                print(f"skip {run.name} (no {kind})")
                continue

            cmd = [
                sys.executable, "-m", "kantorl", "milestones", str(checkpoint),
                args.rom_path,
                "--seeds", *[str(s) for s in args.seeds],
                "--budget", str(args.budget),
                "--stochastic",
                "--out", str(eval_dir),
            ]
            if args.save_state:
                cmd += ["--save-state", args.save_state]
            if args.map_player_marker:
                cmd.append("--map-player-marker")
            print(f"eval {run.name} <- {checkpoint.name}", flush=True)
            subprocess.run(cmd, check=False)

    # ---------------------------------------------------------------- summary
    by_tier: dict[str, list[tuple[str, dict[str, int], int]]] = {}
    for run in runs:
        path = run / ("eval_best" if args.best else "eval") / "milestones.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        episodes = report["episodes"]
        by_tier.setdefault(tier_of(run), []).append(
            (run.name, reach_counts(report), len(episodes))
        )

    if not by_tier:
        print("No reports to summarise.")
        return 1

    for tier, entries in sorted(by_tier.items()):
        print(f"\n== {tier}  ({len(entries)} training run(s), "
              f"{entries[0][2]} episodes each)")
        header = f"   {'milestone':<20} " + " ".join(
            f"{name.replace(tier + '_', ''):>10}" for name, _, _ in entries
        ) + "   mean [min-max]"
        print(header)
        for label in KEY_MILESTONES:
            rates = [counts.get(label, 0) / n for _, counts, n in entries]
            if not any(rates):
                continue
            cells = " ".join(f"{c.get(label, 0)}/{n:<8}" for _, c, n in entries)
            mean = sum(rates) / len(rates)
            print(f"   {label:<20} {cells}   {mean:.2f} "
                  f"[{min(rates):.2f}-{max(rates):.2f}]")
        badges = [
            max((e["badges"] for e in json.loads(
                (Path('runs') / name / 'eval' / 'milestones.json').read_text()
            )["episodes"]), default=0)
            for name, _, _ in entries
        ]
        print(f"   {'badges (max)':<20} " + " ".join(f"{b:>10}" for b in badges))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
