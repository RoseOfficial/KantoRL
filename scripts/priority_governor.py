"""Keep training at low priority while someone is using the machine.

Windows priority classes are static: a process set to BelowNormal stays there
until something changes it, even after the game that prompted it has exited.
The previous attempt used a watchdog that re-applied BelowNormal every 20
seconds, so the run never went back to full speed when the machine went idle.

This governor is state-dependent. It reads the user's idle time and:

    user active  -> BELOW_NORMAL (training yields to whatever you are doing)
    user idle    -> NORMAL      (training uses the whole machine again)

Only processes belonging to the training run are touched: the one whose command
line mentions ``kantorl`` and all of its children (the emulator workers).

Usage:
    python scripts/priority_governor.py --hours 6
    python scripts/priority_governor.py --hours 6 --idle-seconds 120 --interval 15

Stop it with Ctrl+C; it leaves every process at NORMAL on the way out.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from ctypes import wintypes

import psutil


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


# Declare signatures explicitly. Without argtypes, ctypes passes the struct
# pointer as a 32-bit int on 64-bit Windows and the call silently returns 0,
# which reads as "the user is always active" and never restores priority.
_USER32 = ctypes.WinDLL("user32", use_last_error=True)
_USER32.GetLastInputInfo.argtypes = [ctypes.POINTER(LASTINPUTINFO)]
_USER32.GetLastInputInfo.restype = wintypes.BOOL

_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
_KERNEL32.GetTickCount.argtypes = []
_KERNEL32.GetTickCount.restype = wintypes.DWORD


def idle_seconds() -> float:
    """Seconds since the last keyboard or mouse input, on Windows."""
    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if not _USER32.GetLastInputInfo(ctypes.byref(info)):
        raise OSError(ctypes.get_last_error(), "GetLastInputInfo failed")
    now = _KERNEL32.GetTickCount()
    elapsed_ms = (now - info.dwTime) & 0xFFFFFFFF
    return elapsed_ms / 1000.0


def training_processes(marker: str) -> list[psutil.Process]:
    """The training run's process and its emulator workers.

    Matches only ``python -m <marker> ...``. A looser match also caught
    unrelated shells whose command text merely contained the repo path.
    """
    found: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        is_entrypoint = (
            len(cmdline) >= 3
            and cmdline[1] == "-m"
            and cmdline[2] == marker
        )
        if is_entrypoint:
            found.append(proc)
            try:
                found.extend(proc.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return found


def set_nice(procs: list[psutil.Process], value: int) -> int:
    changed = 0
    for proc in procs:
        try:
            if proc.nice() != value:
                proc.nice(value)
                changed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument(
        "--idle-seconds", type=float, default=120.0,
        help="Idle time before training is restored to normal priority",
    )
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--marker", default="kantorl")
    args = parser.parse_args()

    if sys.platform != "win32":
        print("This governor is Windows-only.")
        return 1

    deadline = time.time() + args.hours * 3600
    state: str | None = None
    print(f"governing for {args.hours:g}h; idle threshold {args.idle_seconds:g}s", flush=True)

    try:
        while time.time() < deadline:
            procs = training_processes(args.marker)
            if not procs:
                if state != "none":
                    print("no training processes found; waiting", flush=True)
                    state = "none"
                time.sleep(args.interval)
                continue

            idle = idle_seconds()
            wanted = "idle" if idle >= args.idle_seconds else "active"
            if wanted != state:
                if wanted == "idle":
                    changed = set_nice(procs, psutil.NORMAL_PRIORITY_CLASS)
                    print(f"[{time.strftime('%H:%M:%S')}] idle {idle:.0f}s "
                          f"-> NORMAL ({changed} processes)", flush=True)
                else:
                    changed = set_nice(procs, psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    print(f"[{time.strftime('%H:%M:%S')}] active "
                          f"-> BELOW_NORMAL ({changed} processes)", flush=True)
                state = wanted
            time.sleep(args.interval)
    finally:
        # Never leave a run throttled because the governor exited.
        leftover = training_processes(args.marker)
        restored = set_nice(leftover, psutil.NORMAL_PRIORITY_CLASS)
        print(f"exiting; restored {restored} processes to NORMAL", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
