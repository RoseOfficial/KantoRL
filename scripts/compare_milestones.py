"""Compare milestone reports from several checkpoints side by side.

The per-run report answers "how far did this checkpoint get". The question the
tier experiment actually asks is comparative: which observations are load
bearing. That needs one table with each milestone on a row and each checkpoint
on a column, which is what this prints.

Usage:
    python scripts/compare_milestones.py runs/t0_pure/eval runs/t1_minimal/eval
    python scripts/compare_milestones.py --glob "runs/t*/eval/milestones.json"
    python scripts/compare_milestones.py runs/t2_pragmatic/eval --csv out.csv

Each argument may be a ``milestones.json`` file or a directory to search for
one. Milestones reached at step 0 are marked with ``*``: they hold before the
agent acts at all (the starting map, for instance) and so say nothing about
progress.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Report order: badges first, then story events, then maps.
_KIND_ORDER = {"badge": 0, "event": 1, "map": 2}


def find_reports(paths: list[str], pattern: str | None) -> list[Path]:
    """Resolve CLI arguments into a list of report JSON paths."""
    found: list[Path] = []
    if pattern:
        found.extend(sorted(Path().glob(pattern)))
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            candidate = path / "milestones.json"
            if candidate.is_file():
                found.append(candidate)
            else:
                found.extend(sorted(path.rglob("milestones.json")))
        else:
            raise SystemExit(f"No such report or directory: {path}")

    if not found:
        raise SystemExit("No milestones.json found in the given paths.")
    return found


def label_for(path: Path, report: dict) -> str:
    """Short column name: the run directory if there is one, else the tier."""
    if path.parent.name == "eval":
        return path.parent.parent.name
    if path.parent.name:
        return path.parent.name
    return "+".join(report.get("obs_components", [])) or path.stem


def reach_cell(row: dict) -> str:
    """Render one milestone for one checkpoint as ``rate (median step)``."""
    reached = row.get("reached_count", 0)
    total = row.get("total_runs", 0)
    if not reached:
        return "-"

    rate = f"{reached}/{total}"
    median = row.get("median_first_step")
    if median is None:
        return rate

    step = f"{median:,.0f}"
    if median == 0:
        step += "*"
    return f"{rate} @{step}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", help="milestones.json files or dirs")
    parser.add_argument("--glob", default=None, help="Glob for report files")
    parser.add_argument("--csv", default=None, help="Write the table to this CSV")
    args = parser.parse_args()

    paths = find_reports(args.reports, args.glob)

    columns: list[tuple[str, dict[str, dict]]] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = {row["milestone_key"]: row for row in report.get("summary", [])}
        columns.append((label_for(path, report), rows))

    # Union of milestones, ordered by kind then index so tiers line up.
    keys: dict[str, dict] = {}
    for _, rows in columns:
        for key, row in rows.items():
            keys.setdefault(key, row)

    def sort_key(item: tuple[str, dict]) -> tuple:
        row = item[1]
        return (_KIND_ORDER.get(row.get("kind", ""), 9), row.get("index", 0))

    ordered = sorted(keys.items(), key=sort_key)

    names = [name for name, _ in columns]
    label_width = max([len(row.get("label", "")) for _, row in ordered] + [9])
    col_width = max([len(name) for name in names] + [8])

    header = f"{'milestone':<{label_width}} " + " ".join(
        f"{name:>{col_width}}" for name in names
    )
    print(header)
    print("-" * len(header))
    for key, row in ordered:
        cells = [reach_cell(rows.get(key, {})) for _, rows in columns]
        print(
            f"{row.get('label', key):<{label_width}} "
            + " ".join(f"{cell:>{col_width}}" for cell in cells)
        )

    if any(
        (rows.get(key, {}).get("median_first_step") == 0)
        for key, _ in ordered
        for _, rows in columns
    ):
        print("\n* reached at step 0, i.e. already true before the agent acts.")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["milestone_key", "label", *names])
            for key, row in ordered:
                writer.writerow(
                    [
                        key,
                        row.get("label", key),
                        *[
                            rows.get(key, {}).get("reach_rate", "")
                            for _, rows in columns
                        ],
                    ]
                )
        print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
