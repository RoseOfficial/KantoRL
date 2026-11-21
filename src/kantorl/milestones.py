"""Milestone evaluation for trained KantoRL policies.

Training logs tell you what the reward function was paid for. They do not tell
you *when the agent first reached a real game milestone*. That distinction is
the whole point of running tiers: a policy can farm event flags and item
pickups without ever leaving Pallet Town, and its reward curve will look
healthy while it does.

This module runs a checkpoint under a fixed seed and step budget, watches the
game state every step, and records the first step at which each milestone was
reached:

* ``badge:N``  -- the Nth gym badge
* ``event:I``  -- curated story event bit ``I`` (see ``data/events.json``)
* ``map:M``    -- first visit to map ``M``

Milestones are read from RAM for *evaluation only*. Nothing here is fed to the
policy, so the same harness measures a pixels-only tier and a RAM-rich tier
identically. That is what makes the numbers comparable.
"""

from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kantorl.checkpoints import (
    assert_observation_compatible,
    env_kwargs_from_model,
    is_recurrent,
    load_model,
)
from kantorl.config import KantoConfig
from kantorl.env import KantoRedEnv
from kantorl.global_map import get_map_name

__all__ = [
    "BADGE_LABELS",
    "CORE_MILESTONE_KEYS",
    "EpisodeResult",
    "Milestone",
    "MilestoneHit",
    "MilestoneReport",
    "MilestoneTracker",
    "evaluate_milestones",
    "load_event_labels",
    "warn_if_deterministic_seeds",
    "write_episodes_csv",
    "write_report",
    "write_report_json",
    "write_summary_csv",
]

# =============================================================================
# MILESTONE CATALOG
# =============================================================================

# The eight gym badges, in badge-number order (1 = Boulder).
BADGE_LABELS: tuple[str, ...] = (
    "Boulder Badge",
    "Cascade Badge",
    "Thunder Badge",
    "Rainbow Badge",
    "Soul Badge",
    "Marsh Badge",
    "Volcano Badge",
    "Earth Badge",
)

# Early-game chain every tier is measured against, by event bit index.
# These always appear in the summary, even when a policy never reaches them:
# "never reached" is the result that matters most for a naive tier.
#
# Indices are taken from pret/pokered `constants/event_constants.asm` and
# verified empirically: a save state holding a starter has bit 34 set. The
# previous indices were hand-guessed and watched unrelated bits -- bit 50 is
# not "Got Starter", so that milestone could never fire.
_CORE_EVENT_INDICES: tuple[int, ...] = (
    0,     # Followed Oak into the lab
    33,    # Oak asked which starter to take
    34,    # Got starter
    36,    # Got Pokeballs from Oak
    37,    # Got Pokedex
    57,    # Picked up Oak's Parcel
    56,    # Delivered Oak's Parcel
    1317,  # Beat the Route 22 rival (first battle)
)

# Stable key set that is always reported.
CORE_MILESTONE_KEYS: tuple[str, ...] = tuple(
    [f"badge:{number}" for number in range(1, len(BADGE_LABELS) + 1)]
    + [f"event:{index}" for index in _CORE_EVENT_INDICES]
)

_EVENT_LABELS: dict[int, str] | None = None


def load_event_labels() -> dict[int, str]:
    """Load curated event-bit names from ``data/events.json`` (cached)."""
    global _EVENT_LABELS
    if _EVENT_LABELS is None:
        path = Path(__file__).parent / "data" / "events.json"
        if not path.exists():  # pragma: no cover - packaged data should exist
            _EVENT_LABELS = {}
        else:
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
            _EVENT_LABELS = {int(key): value for key, value in raw.items()}
    return _EVENT_LABELS


@dataclass(frozen=True)
class Milestone:
    """One reachable game milestone.

    Attributes:
        key: Stable identifier used in JSON/CSV, e.g. ``"event:3"``.
        kind: ``"badge"``, ``"event"`` or ``"map"``.
        index: Badge number (1-based), event bit index, or map ID.
        label: Human-readable name.
    """

    key: str
    kind: str
    index: int
    label: str

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "kind": self.kind, "index": self.index, "label": self.label}


def badge_milestone(number: int) -> Milestone:
    """Build the milestone for badge ``number`` (1-based)."""
    if not 1 <= number <= len(BADGE_LABELS):
        raise ValueError(f"Badge number must be 1..{len(BADGE_LABELS)}, got {number}")
    return Milestone(
        key=f"badge:{number}",
        kind="badge",
        index=number,
        label=BADGE_LABELS[number - 1],
    )


def event_milestone(index: int) -> Milestone:
    """Build the milestone for event bit ``index``."""
    label = load_event_labels().get(index, f"Event {index}")
    return Milestone(key=f"event:{index}", kind="event", index=index, label=label)


def map_milestone(map_id: int) -> Milestone:
    """Build the milestone for first visit to ``map_id``."""
    return Milestone(
        key=f"map:{map_id}",
        kind="map",
        index=map_id,
        label=get_map_name(map_id),
    )


@dataclass(frozen=True)
class MilestoneHit:
    """A milestone reached for the first time at ``step`` in an episode."""

    milestone: Milestone
    step: int


# =============================================================================
# TRACKER
# =============================================================================


class MilestoneTracker:
    """Record the first step at which each milestone was reached.

    The tracker is deliberately ignorant of how the policy observes the game:
    it consumes the environment's ``info`` dict, which is populated from RAM
    regardless of the observation tier.
    """

    def __init__(self) -> None:
        self._first_step: dict[str, int] = {}
        self._catalog: dict[str, Milestone] = {}
        self._events_baselined = False

    def reset(self) -> None:
        """Clear per-episode tracking (catalog included)."""
        self._first_step.clear()
        self._catalog.clear()
        self._events_baselined = False

    @property
    def first_steps(self) -> dict[str, int]:
        """Mapping of milestone key to the step it was first reached."""
        return dict(self._first_step)

    @property
    def catalog(self) -> dict[str, Milestone]:
        """Mapping of milestone key to milestone metadata."""
        return dict(self._catalog)

    def _record(self, milestone: Milestone, step: int) -> MilestoneHit | None:
        if milestone.key in self._first_step:
            return None
        self._first_step[milestone.key] = step
        self._catalog[milestone.key] = milestone
        return MilestoneHit(milestone=milestone, step=step)

    def update(self, info: dict[str, Any]) -> list[MilestoneHit]:
        """Fold one environment ``info`` dict into the tracker.

        Args:
            info: The info dict returned by ``env.step`` / ``env.reset``.

        Returns:
            Milestones reached for the first time by this step, in a stable
            order: badges, then events, then maps.
        """
        step = int(info.get("step", 0))
        hits: list[MilestoneHit] = []

        badge_count = int(info.get("badges", 0) or 0)
        for number in range(1, badge_count + 1):
            hit = self._record(badge_milestone(number), step)
            if hit is not None:
                hits.append(hit)

        event_indices = [int(i) for i in (info.get("new_event_indices") or ())]
        event_count = int(info.get("events", 0) or 0)
        if not self._events_baselined and event_count > 0:
            self._events_baselined = True
            if step == 0:
                # The episode began mid-game (a save state), so these flags are
                # progress the run inherited rather than earned. Report them at
                # step 0, which the report keeps distinct from "never reached".
                event_indices = [
                    int(i) for i in (info.get("event_indices_set") or ())
                ]
            else:
                # Starting a new game sets a block of flags at once (0, 25, 32,
                # 33, 34, 35, 39 and 960 on Red UE). They arrive before the
                # player can act and are not achievements, but they look exactly
                # like one to a difference detector. Treat the first batch as
                # the initial state, so nothing in it counts as progress.
                event_indices = []

        for index in event_indices:
            hit = self._record(event_milestone(index), step)
            if hit is not None:
                hits.append(hit)

        map_id = info.get("map_id")
        if map_id is not None:
            hit = self._record(map_milestone(int(map_id)), step)
            if hit is not None:
                hits.append(hit)

        return hits


# =============================================================================
# REPORT
# =============================================================================


@dataclass
class EpisodeResult:
    """Outcome of one evaluation episode."""

    episode: int
    seed: int
    steps: int
    reward: float
    badges: int
    events: int
    explore_tiles: int
    party_count: int
    level_sum: int = 0
    max_level: int = 0
    first_steps: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "seed": self.seed,
            "steps": self.steps,
            "reward": self.reward,
            "badges": self.badges,
            "events": self.events,
            "explore_tiles": self.explore_tiles,
            "party_count": self.party_count,
            "level_sum": self.level_sum,
            "max_level": self.max_level,
            "first_steps": dict(self.first_steps),
        }


@dataclass
class MilestoneReport:
    """Milestone results for one checkpoint across a set of seeds."""

    checkpoint: str
    obs_components: tuple[str, ...]
    budget: int
    deterministic: bool
    episodes: list[EpisodeResult]
    catalog: dict[str, Milestone]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def milestone_keys(self) -> list[str]:
        """Catalog keys ordered by kind, then index."""
        order = {"badge": 0, "event": 1, "map": 2}
        return sorted(
            self.catalog,
            key=lambda key: (order.get(self.catalog[key].kind, 9), self.catalog[key].index),
        )

    def reached_count(self, key: str) -> int:
        """How many episodes reached ``key``."""
        return sum(1 for episode in self.episodes if key in episode.first_steps)

    def summary_rows(self) -> list[dict[str, Any]]:
        """One row per milestone: reach rate and first-step statistics.

        ``median_first_step`` is only over episodes that reached the milestone.
        It is ``None`` when it was never reached, which is different from
        "reached at step 0" and must stay distinct in the output.
        """
        total = len(self.episodes)
        rows: list[dict[str, Any]] = []

        for key in self.milestone_keys():
            milestone = self.catalog[key]
            steps = [
                episode.first_steps[key]
                for episode in self.episodes
                if key in episode.first_steps
            ]
            reached = len(steps)
            rows.append(
                {
                    "milestone_key": key,
                    "kind": milestone.kind,
                    "index": milestone.index,
                    "label": milestone.label,
                    "reached_count": reached,
                    "total_runs": total,
                    "reach_rate": reached / total if total else 0.0,
                    "median_first_step": float(np.median(steps)) if steps else None,
                    "min_first_step": min(steps) if steps else None,
                }
            )

        return rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "obs_components": list(self.obs_components),
            "budget": self.budget,
            "deterministic": self.deterministic,
            "episodes": [episode.as_dict() for episode in self.episodes],
            "catalog": {
                key: self.catalog[key].as_dict() for key in self.milestone_keys()
            },
            "summary": self.summary_rows(),
        }

    def format_summary(self, max_rows: int | None = None) -> str:
        """Render a fixed-width table of the summary rows."""
        rows = self.summary_rows()
        if max_rows is not None:
            rows = rows[:max_rows]
        if not rows:
            return "(no milestones reached)"

        header = f"{'milestone':<28} {'reached':>9} {'rate':>6} {'median step':>12}"
        lines = [header, "-" * len(header)]
        for row in rows:
            median = row["median_first_step"]
            median_text = "-" if median is None else f"{median:.0f}"
            lines.append(
                f"{row['label'][:28]:<28} "
                f"{row['reached_count']:>4}/{row['total_runs']:<4} "
                f"{row['reach_rate']:>6.0%} "
                f"{median_text:>12}"
            )
        return "\n".join(lines)


# =============================================================================
# EVALUATION LOOP
# =============================================================================


def _act(
    model: Any,
    obs: dict[str, np.ndarray],
    state: Any,
    episode_start: np.ndarray,
    recurrent: bool,
    deterministic: bool,
) -> tuple[Any, Any, np.ndarray]:
    """Select one action, threading LSTM state when the policy is recurrent."""
    if recurrent:
        action, state = model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return action, state, np.zeros((1,), dtype=bool)
    action, _ = model.predict(obs, deterministic=deterministic)
    return action, state, episode_start


def warn_if_deterministic_seeds(deterministic: bool, seeds: Sequence[int]) -> None:
    """Warn when deterministic evaluation is asked for several seeds.

    An argmax policy on a deterministic emulator from a fixed start state
    produces byte-identical episodes, so N seeds are N copies of one run and
    look like replication when they are not. Call before the eval loop.
    """
    if deterministic and len(seeds) > 1:
        warnings.warn(
            "deterministic=True makes every seed replay the same episode: the "
            "policy is argmax and the emulator is deterministic. Seeds only "
            "vary action sampling, so use --stochastic for a real spread.",
            UserWarning,
            stacklevel=2,
        )


def evaluate_milestones(
    checkpoint_path: str | Path,
    rom_path: str | Path,
    *,
    seeds: Sequence[int] = (0,),
    budget: int = 163_840,
    deterministic: bool = True,
    render: bool = False,
    reward_fn: str = "default",
    base_config: KantoConfig | None = None,
    verbose: bool = True,
) -> MilestoneReport:
    """Run a checkpoint and record when each milestone was first reached.

    Every call uses the same seeds and the same step budget, so reports from
    different tiers are directly comparable. The observation layout is taken
    from the checkpoint itself; ``base_config`` only supplies settings that
    cannot be recovered from it (``action_freq``, ``headless``, ...).

    Args:
        checkpoint_path: Path to a ``.zip`` checkpoint (PPO or RecurrentPPO).
        rom_path: Path to the Pokemon Red ROM used for training.
        seeds: Episode seeds. Each seed is one episode; identical seeds across
            checkpoints are what make the comparison fair.
        budget: Maximum environment steps per episode. Truncation is enforced
            here, so a checkpoint cannot get a longer episode than another.
        deterministic: Use argmax actions (True) or sample (False).
        render: Open the SDL2 window while evaluating.
        reward_fn: Reward function name; affects only the reported reward sum,
            never the milestones.
        base_config: Optional config for non-observable settings such as
            ``action_freq``. Observation fields are overridden by the checkpoint.
        verbose: Print per-episode progress.

    Returns:
        A :class:`MilestoneReport` with per-episode first-step timestamps.

    Raises:
        ValueError: If ``seeds`` is empty or ``budget`` is not positive.
    """
    if not seeds:
        raise ValueError("seeds must contain at least one episode seed")
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if deterministic and len(seeds) > 1:
        warn_if_deterministic_seeds(deterministic, seeds)

    model = load_model(checkpoint_path)

    # The checkpoint is authoritative for anything shape-dependent: frame
    # stack, screen size and the observation tier. Getting this wrong is a
    # tensor-shape crash, not a subtle bug, so infer rather than ask.
    inferred = env_kwargs_from_model(model)
    config = base_config or KantoConfig(rom_path=str(rom_path))
    config.rom_path = str(rom_path)
    config.max_steps = int(budget)
    config.obs_components = tuple(inferred["obs_components"])
    if "frame_stacks" in inferred:
        config.frame_stacks = int(inferred["frame_stacks"])
    if "screen_size" in inferred:
        config.screen_size = inferred["screen_size"]

    env = KantoRedEnv(
        config=config,
        reward_fn=reward_fn,
        render_mode="human" if render else None,
    )

    try:
        assert_observation_compatible(model, env)
        recurrent = is_recurrent(model)
        episodes: list[EpisodeResult] = []

        for episode_index, seed in enumerate(seeds):
            obs, info = env.reset(seed=int(seed))

            tracker = MilestoneTracker()
            tracker.update(info)

            state: Any = None
            episode_start = np.ones((1,), dtype=bool)
            total_reward = 0.0
            steps = 0

            while steps < budget:
                action, state, episode_start = _act(
                    model,
                    obs,
                    state,
                    episode_start,
                    recurrent,
                    deterministic,
                )
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                steps += 1
                tracker.update(info)

                if terminated or truncated:
                    break

            result = EpisodeResult(
                episode=episode_index + 1,
                seed=int(seed),
                steps=steps,
                reward=total_reward,
                badges=int(info.get("badges", 0) or 0),
                events=int(info.get("events", 0) or 0),
                explore_tiles=int(info.get("explore_tiles", 0) or 0),
                party_count=int(info.get("party_count", 0) or 0),
                level_sum=int(info.get("level_sum", 0) or 0),
                max_level=int(info.get("max_level", 0) or 0),
                first_steps=tracker.first_steps,
            )
            episodes.append(result)

            if verbose:
                print(
                    f"seed {seed}: {steps} steps, reward {total_reward:.1f}, "
                    f"badges {result.badges}, events {result.events}, "
                    f"tiles {result.explore_tiles}, "
                    f"milestones {len(result.first_steps)}"
                )

        # Merge catalogs across episodes so a milestone reached in any run is
        # described once, even if it appears late in the report. Core
        # milestones are seeded first so "never reached" is an explicit row
        # rather than a missing one.
        catalog: dict[str, Milestone] = {
            key: _milestone_from_key(key) for key in CORE_MILESTONE_KEYS
        }
        for episode in episodes:
            for key in episode.first_steps:
                if key not in catalog:
                    catalog[key] = _milestone_from_key(key)

        return MilestoneReport(
            checkpoint=str(checkpoint_path),
            obs_components=tuple(config.obs_components),
            budget=int(budget),
            deterministic=deterministic,
            episodes=episodes,
            catalog=catalog,
        )
    finally:
        env.close()


def _milestone_from_key(key: str) -> Milestone:
    """Rebuild a milestone from its stable key."""
    kind, _, raw_index = key.partition(":")
    index = int(raw_index)
    if kind == "badge":
        return badge_milestone(index)
    if kind == "event":
        return event_milestone(index)
    if kind == "map":
        return map_milestone(index)
    raise ValueError(f"Unknown milestone key: {key!r}")


# =============================================================================
# SERIALIZATION
# =============================================================================


def write_report_json(report: MilestoneReport, path: str | Path) -> Path:
    """Write the full report (episodes + catalog + summary) as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)
    return out


_EPISODE_COLUMNS = (
    "run_id",
    "checkpoint",
    "obs_components",
    "seed",
    "episode",
    "steps",
    "reward",
    "badges",
    "events",
    "explore_tiles",
    "party_count",
    "level_sum",
    "max_level",
    "milestone_key",
    "milestone_kind",
    "milestone_label",
    "first_step",
)


def write_episodes_csv(report: MilestoneReport, path: str | Path) -> Path:
    """Write a long-format CSV: one row per (episode, reached milestone)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tier = "+".join(report.obs_components)

    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_EPISODE_COLUMNS)
        writer.writeheader()
        for episode in report.episodes:
            for key, step in sorted(episode.first_steps.items()):
                milestone = _milestone_from_key(key)
                writer.writerow(
                    {
                        "run_id": Path(report.checkpoint).stem,
                        "checkpoint": report.checkpoint,
                        "obs_components": tier,
                        "seed": episode.seed,
                        "episode": episode.episode,
                        "steps": episode.steps,
                        "reward": f"{episode.reward:.4f}",
                        "badges": episode.badges,
                        "events": episode.events,
                        "explore_tiles": episode.explore_tiles,
                        "party_count": episode.party_count,
                        "level_sum": episode.level_sum,
                        "max_level": episode.max_level,
                        "milestone_key": key,
                        "milestone_kind": milestone.kind,
                        "milestone_label": milestone.label,
                        "first_step": step,
                    }
                )
    return out


_SUMMARY_COLUMNS = (
    "run_id",
    "obs_components",
    "milestone_key",
    "kind",
    "index",
    "label",
    "reached_count",
    "total_runs",
    "reach_rate",
    "median_first_step",
    "min_first_step",
)


def write_summary_csv(report: MilestoneReport, path: str | Path) -> Path:
    """Write one row per milestone with reach rate and first-step statistics."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tier = "+".join(report.obs_components)

    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in report.summary_rows():
            writer.writerow(
                {
                    "run_id": Path(report.checkpoint).stem,
                    "obs_components": tier,
                    **row,
                }
            )
    return out


def write_report(
    report: MilestoneReport,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write JSON, episode CSV and summary CSV into ``out_dir``."""
    directory = Path(out_dir)
    return {
        "json": write_report_json(report, directory / "milestones.json"),
        "episodes_csv": write_episodes_csv(report, directory / "milestones_episodes.csv"),
        "summary_csv": write_summary_csv(report, directory / "milestones_summary.csv"),
    }
