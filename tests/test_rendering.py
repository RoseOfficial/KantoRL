"""Regression tests for the emulator rendering contract.

Bug this protects: ``_execute_action`` advanced the emulator with
``tick(n, render=False)`` for every frame. PyBoy 2.9 does not update the screen
buffer when ``render`` is False, so ``_get_screen_gray()`` read a frozen frame
and every "screens" observation was identical. Measured over 120 steps the
screen changed 0 times; a pixel-only policy was effectively blind.

The contract is: the final frame of each action must render. Rendering only
that frame costs ~1% because PyBoy renders once per multi-frame tick.
"""

from __future__ import annotations

from kantorl.config import KantoConfig
from kantorl.env import KantoRedEnv


class FakePyBoy:
    """Records tick calls instead of emulating."""

    def __init__(self) -> None:
        self.ticks: list[tuple[int, bool]] = []
        self.inputs: list[object] = []

    def send_input(self, event: object) -> None:
        self.inputs.append(event)

    def tick(self, count: int = 1, render: bool = True) -> None:
        self.ticks.append((count, render))


def make_env() -> KantoRedEnv:
    """Build an env without touching the ROM (the emulator is lazy)."""
    return KantoRedEnv(config=KantoConfig(rom_path="unused.gb"))


def test_last_frame_of_an_action_renders():
    env = make_env()
    fake = FakePyBoy()
    env._pyboy = fake  # inject before the lazy property runs

    env._execute_action(1)  # UP

    assert fake.ticks, "no frames were advanced"
    assert fake.ticks[-1][1] is True, "final frame must render or screens freeze"
    assert sum(c for c, _ in fake.ticks) == env.config.action_freq


def test_noop_still_renders_the_final_frame():
    env = make_env()
    fake = FakePyBoy()
    env._pyboy = fake

    env._execute_action(0)  # NOOP: no button, but time must still advance

    assert fake.ticks[-1][1] is True
    assert sum(c for c, _ in fake.ticks) == env.config.action_freq


def test_frames_before_the_last_do_not_render():
    """Only one render per action; rendering every frame costs throughput."""
    env = make_env()
    fake = FakePyBoy()
    env._pyboy = fake

    env._execute_action(5)  # A

    assert sum(1 for _, render in fake.ticks if render) == 1


def test_fast_mode_is_on_by_default():
    """mGBA-class speed without it; PyBoy's optimised path is ~3.2x."""
    assert KantoConfig().fast_mode is True
