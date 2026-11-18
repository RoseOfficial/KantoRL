"""Queue tier training runs across seeds, one after another.

Replication is the thing the single-seed results need, and doing it by hand is
how you end up with three runs on different settings. This runs a fixed matrix
of (tier, seed) with identical arguments, skips anything already finished, and
prints a summary at the end.

Runs are sequential on purpose: parallel training runs compete for the same
cores, and measured throughput roughly halves when they do.

Usage:
    python scripts/run_tiers.py "C:/roms/Pokemon Red.gb" --seeds 43 44
    python scripts/run_tiers.py ROM --tiers t0_pure t2_pragmatic --seeds 43 44
    python scripts/run_tiers.py ROM --steps 20000000 --save-state states/bulba_start.state
    python scripts/run_tiers.py ROM --seeds 43 44 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TIERS = ("t0_pure", "t2_pragmatic")


def finished_steps(checkpoint_dir: Path) -> int:
    """Highest timestep among saved checkpoints, or 0 if none."""
    best = 0
    if not checkpoint_dir.is_dir():
        return 0
    for path in checkpoint_dir.glob("model_*.zip"):
        try:
            best = max(best, int(path.stem.removeprefix("model_")))
        except ValueError:
            continue
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom_path")
    parser.add_argument("--tiers", nargs="+", default=list(DEFAULT_TIERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--tag", default="", help="Session tag, e.g. _mod")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not Path(args.rom_path).is_file():
        print(f"ROM not found: {args.rom_path}")
        return 1

    plan = []
    for tier in args.tiers:
        for seed in args.seeds:
            session = Path("runs") / f"{tier}_{args.steps // 1_000_000}M{args.tag}_s{seed}"
            done = finished_steps(session / "checkpoints")
            plan.append((tier, seed, session, done))

    width = max(len(str(s)) for _, _, s, _ in plan) + 2
    print(f"{'session':<{width}} {'have':>12} {'target':>12}  status")
    print("-" * (width + 40))
    for _, _, session, done in plan:
        status = "skip (done)" if done >= args.steps else ("resume" if done else "run")
        print(f"{str(session):<{width}} {done:>12,} {args.steps:>12,}  {status}")

    if args.dry_run:
        return 0

    failures = []
    started = time.perf_counter()
    for tier, seed, session, done in plan:
        if done >= args.steps:
            continue

        cmd = [
            sys.executable, "-m", "kantorl", "tier", "run", tier, args.rom_path,
            "--session", str(session),
            "--steps", str(args.steps),
            "--seed", str(seed),
            "--no-progress",
        ]
        if args.save_state:
            cmd += ["--save-state", args.save_state]

        print(f"\n=== {session} (seed {seed}) ===", flush=True)
        run_start = time.perf_counter()
        result = subprocess.run(cmd, check=False)
        elapsed = time.perf_counter() - run_start
        if result.returncode != 0:
            failures.append((str(session), result.returncode))
            print(f"    FAILED after {elapsed / 60:.1f} min (exit {result.returncode})")
        else:
            print(f"    finished in {elapsed / 60:.1f} min")

    total = (time.perf_counter() - started) / 60
    print(f"\nTotal wall clock: {total:.1f} min")
    if failures:
        print("Failures:")
        for session, code in failures:
            print(f"  {session} (exit {code})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
