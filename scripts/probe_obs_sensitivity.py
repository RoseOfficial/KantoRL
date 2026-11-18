"""Measure how much each observation component actually affects the policy.

A null result between tiers is only meaningful if the richer tier's policy uses
the extra inputs. A 2,560-bit event vector into a small MLP can be silently
ignored, in which case the RAM-rich tier is functionally the pixel tier and any
comparison between them measures nothing.

For each component: collect real observations, replace that component with
noise, and measure how much the action distribution moves. A component the
policy relies on produces a large KL; one it ignores produces ~0.

Usage:
    python scripts/probe_obs_sensitivity.py runs/t2_pragmatic_20M_s42
    python scripts/probe_obs_sensitivity.py RUN --steps 300 --batch 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_baselines3.common.utils import obs_as_tensor  # noqa: E402

from kantorl.checkpoints import env_kwargs_from_model, load_model  # noqa: E402
from kantorl.config import KantoConfig  # noqa: E402
from kantorl.env import KantoRedEnv  # noqa: E402


def randomise(key: str, value: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace one observation component with plausible noise."""
    shape = value.shape
    if key == "screens":
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    if key == "health":
        return rng.random(shape, dtype=np.float32)
    if key == "level":
        return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if key == "badges":
        return rng.integers(0, 2, size=shape).astype(value.dtype)
    if key == "events":
        return rng.integers(0, 2, size=shape).astype(value.dtype)
    if key == "map":
        return (rng.integers(0, 2, size=shape) * 255).astype(value.dtype)
    if key == "recent_actions":
        return rng.integers(0, 8, size=shape).astype(value.dtype)
    return value


def action_probs(model, obs_batch: dict[str, np.ndarray]):
    tensor = obs_as_tensor(obs_batch, model.device)
    with np.errstate(all="ignore"):
        import torch

        with torch.no_grad():
            probs = model.policy.get_distribution(tensor).distribution.probs
    return probs.cpu().numpy()


def kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0)
    q = np.clip(q, 1e-9, 1.0)
    return np.sum(p * np.log(p / q), axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--rom", default=r"C:\Users\neoga\roms\Pokemon Red.gb")
    parser.add_argument(
        "--save-state", default=r"C:\Users\neoga\roms\states\bulba_start.state"
    )
    parser.add_argument("--steps", type=int, default=250, help="Transitions to collect")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint = args.checkpoint or max(
        (run_dir / "checkpoints").glob("model_*.zip"),
        key=lambda p: int(p.stem.removeprefix("model_")),
    )
    model = load_model(str(checkpoint))
    inferred = env_kwargs_from_model(model)

    cfg = KantoConfig(
        rom_path=args.rom, headless=True,
        save_state_path=args.save_state,
        max_steps=args.steps + 10,
    )
    cfg.obs_components = tuple(inferred["obs_components"])
    cfg.frame_stacks = int(inferred["frame_stacks"])
    cfg.screen_size = inferred["screen_size"]

    env = KantoRedEnv(config=cfg, reward_fn="default")
    obs, _ = env.reset(seed=args.seed)

    collected: dict[str, list[np.ndarray]] = {k: [] for k in obs}
    rng = np.random.default_rng(args.seed)
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=False)
        for key, value in obs.items():
            collected[key].append(np.asarray(value))
        obs, _r, _t, trunc, _i = env.step(int(action))
        if trunc:
            obs, _ = env.reset(seed=args.seed + 1)
    env.close()

    batch = {k: np.stack(v) for k, v in collected.items() if v}
    base = action_probs(model, batch)
    base_top = base.argmax(axis=1)

    print(f"checkpoint : {checkpoint.name}")
    print(f"components : {', '.join(batch)}")
    print(f"transitions: {len(base_top)}   (chance KL vs uniform = "
          f"{np.log(base.shape[1]):.3f} nats)\n")

    print(f"{'component':<16} {'mean KL':>9} {'p95 KL':>8} {'top-1 changed':>14}")
    print("-" * 51)
    for key in batch:
        perturbed = {k: v.copy() for k, v in batch.items()}
        perturbed[key] = randomise(key, batch[key], rng)
        probs = action_probs(model, perturbed)
        kl = kl_rows(base, probs)
        changed = float((probs.argmax(axis=1) != base_top).mean())
        print(f"{key:<16} {kl.mean():>9.4f} {np.percentile(kl, 95):>8.4f} "
              f"{changed:>13.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
