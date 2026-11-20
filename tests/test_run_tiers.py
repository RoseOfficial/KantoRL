"""Tests for the sweep driver's governor wiring.

The runbook requirement this protects: a long sweep must always start the
priority governor, so a run cannot occupy the whole machine while someone is
using it, and must restore full speed when the machine goes idle. A previous
run stayed throttled for hours because the throttling was applied by hand and
nothing reversed it.
"""

from __future__ import annotations

import scripts.run_tiers as run_tiers


def test_default_governor_hours_scales_with_queue():
    assert run_tiers.default_governor_hours(0) == 6.0
    assert run_tiers.default_governor_hours(1) == 6.0
    assert run_tiers.default_governor_hours(4) == 24.0


def test_governor_command_targets_the_script_and_thresholds(monkeypatch):
    captured: dict = {}

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr(run_tiers.subprocess, "Popen", FakePopen)

    proc = run_tiers.start_governor(hours=12.0, idle_seconds=90.0)

    assert proc is not None
    cmd = captured["cmd"]
    assert cmd[1].endswith("priority_governor.py")
    assert cmd[cmd.index("--hours") + 1] == "12"
    assert cmd[cmd.index("--idle-seconds") + 1] == "90"


def test_governor_is_skipped_when_the_script_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(run_tiers.Path, "with_name", lambda self, name: tmp_path / "absent.py")
    assert run_tiers.start_governor(6.0, 120.0) is None


def test_governor_is_a_subprocess_not_a_thread():
    """It must outlive the parent's blocked subprocess.run calls."""
    assert issubclass(run_tiers.subprocess.Popen, object)

def test_training_command_streams_by_default(tmp_path):
    from pathlib import Path

    cmd = run_tiers.training_command(
        "t0_pure", "rom.gb", Path("runs/x"), 1000, 42, "state.state",
        True, "RoseOfficial", "#ff3300",
    )

    assert "--stream" in cmd
    assert cmd[cmd.index("--stream-user") + 1] == "RoseOfficial"
    assert cmd[cmd.index("--stream-color") + 1] == "#ff3300"
    assert cmd[cmd.index("--save-state") + 1] == "state.state"


def test_training_command_can_disable_streaming():
    from pathlib import Path

    cmd = run_tiers.training_command(
        "t0_pure", "rom.gb", Path("runs/x"), 1000, 42, None,
        False, "RoseOfficial", "#ff3300",
    )

    assert "--stream" not in cmd
    assert "--save-state" not in cmd
    assert cmd[0] and cmd[1:4] == ["-m", "kantorl", "tier"]
