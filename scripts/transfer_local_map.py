"""Warm-start a checkpoint with the opt-in ``local_map`` observation.

Adding an observation key changes the extractor's fused width, so a plain
``PPO.load`` cannot resume. This script:

1. Loads the old policy.
2. Builds a new policy whose observation space includes ``local_map``.
3. Copies every shared tensor, including the old columns of the fusion head.
4. Initialises the new head columns small but non-zero, so the local-map
   branch receives gradients immediately while the initial policy stays close
   to the old one.
5. Saves the result with the run's expected ``model_<steps>.zip`` name.

The new run can then resume through the normal ``train.py`` path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from kantorl.config import KantoConfig
from kantorl.env import KantoRedEnv
from kantorl.tiers import load_tier
from kantorl.train import create_policy_kwargs

HEAD_WEIGHTS = (
    "features_extractor.head.0.weight",
    "pi_features_extractor.head.0.weight",
    "vf_features_extractor.head.0.weight",
)
HEAD_BIAS = "features_extractor.head.0.bias"


def _branch_offsets(extractor) -> dict[str, tuple[int, int]]:
    """Start column and width of each branch in the fusion head's input.

    ``ModularExtractor.forward`` concatenates every image branch first, then
    every vector branch, so this must follow the encoder dict order rather
    than the alphabetical observation-space order. Copy by branch, not by
    prefix, or every branch after the new local map shifts by 64 columns.
    """
    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for key, encoder in extractor.image_encoders.items():
        width = int(encoder.features_dim)
        offsets[key] = (offset, width)
        offset += width
    for key, encoder in extractor.vector_encoders.items():
        width = int(encoder[-2].out_features)
        offsets[key] = (offset, width)
        offset += width
    return offsets


def _env_config(
    tier: str,
    rom: str,
    save_state: str | None,
) -> KantoConfig:
    config = load_tier(tier).to_config(rom)
    if save_state:
        config.save_state_path = save_state
    return config


def _make_env(config: KantoConfig) -> DummyVecEnv:
    return DummyVecEnv([lambda: KantoRedEnv(config=config, reward_fn="default")])


def _remap_vecnormalize(
    old_path: str,
    old_config: KantoConfig,
    new_env: DummyVecEnv,
    out_path: Path,
) -> None:
    """Reuse reward-normalization stats on the larger observation space.

    VecNormalize stores the observation space and refuses to wrap a different
    key set, even when ``norm_obs=False``. Rebuild the wrapper around the new
    env and carry over the reward running statistics.
    """
    old_env = _make_env(old_config)
    try:
        old_vec = VecNormalize.load(old_path, old_env)
        new_vec = VecNormalize(
            new_env,
            norm_obs=old_vec.norm_obs,
            norm_reward=old_vec.norm_reward,
            clip_obs=old_vec.clip_obs,
            clip_reward=old_vec.clip_reward,
            gamma=old_vec.gamma,
            epsilon=old_vec.epsilon,
        )
        new_vec.ret_rms = old_vec.ret_rms
        if old_vec.norm_obs:
            new_vec.obs_rms = old_vec.obs_rms
        new_vec.save(str(out_path))
    finally:
        old_env.close()


def _random_observations(space: spaces.Dict) -> dict[str, torch.Tensor]:
    """One random batch shaped like the policy input, for the parity check."""
    batch: dict[str, torch.Tensor] = {}
    for key, subspace in space.spaces.items():
        if isinstance(subspace, spaces.Box) and len(subspace.shape) == 3:
            sample = np.random.randint(0, 256, size=subspace.shape, dtype=np.uint8)
        elif isinstance(subspace, spaces.MultiBinary):
            sample = np.random.randint(0, 2, size=subspace.shape).astype(np.int8)
        elif isinstance(subspace, spaces.MultiDiscrete):
            sample = np.array(
                [np.random.randint(0, int(n)) for n in subspace.nvec],
                dtype=np.int64,
            )
        else:
            sample = np.random.random(subspace.shape).astype(np.float32)
        batch[key] = torch.as_tensor(sample).unsqueeze(0)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--old-tier", default="t2_levelmax_h60_lr5e5")
    parser.add_argument("--new-tier", default="t2_levelmax_h60_lr5e5_lmap")
    parser.add_argument("--old-steps", type=int, required=True)
    parser.add_argument("--out", required=True, help="New run directory")
    parser.add_argument("--rom", required=True)
    parser.add_argument("--save-state", default=None)
    parser.add_argument("--old-vecnormalize", default=None)
    parser.add_argument("--old-first-seen", default=None)
    parser.add_argument("--new-column-std", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.out)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    old_model = PPO.load(args.old_checkpoint, device="cpu")
    old_sd = old_model.policy.state_dict()

    new_config = _env_config(args.new_tier, args.rom, args.save_state)
    new_env = _make_env(new_config)
    new_model = PPO(
        "MultiInputPolicy",
        new_env,
        n_steps=new_config.n_steps,
        batch_size=new_config.batch_size,
        n_epochs=new_config.n_epochs,
        gamma=new_config.gamma,
        gae_lambda=new_config.gae_lambda,
        clip_range=new_config.clip_range,
        target_kl=new_config.target_kl,
        ent_coef=new_config.ent_coef,
        vf_coef=new_config.vf_coef,
        learning_rate=new_config.resume_learning_rate or new_config.learning_rate,
        policy_kwargs=create_policy_kwargs(),
        seed=42,
        device="cpu",
        verbose=0,
    )

    new_sd = new_model.policy.state_dict()
    copied = 0
    skipped: list[str] = []
    for key, value in old_sd.items():
        if key in new_sd and new_sd[key].shape == value.shape:
            new_sd[key] = value.clone()
            copied += 1
        else:
            skipped.append(key)

    old_offsets = _branch_offsets(old_model.policy.features_extractor)
    new_offsets = _branch_offsets(new_model.policy.features_extractor)
    new_keys = [key for key in new_offsets if key not in old_offsets]
    old_width = old_sd[HEAD_WEIGHTS[0]].shape[1]
    new_width = new_sd[HEAD_WEIGHTS[0]].shape[1]
    if new_width <= old_width:
        raise RuntimeError(f"new head width {new_width} is not larger than {old_width}")

    for head_key in HEAD_WEIGHTS:
        old_head = old_sd[head_key]
        new_head = new_sd[head_key]
        for key, (old_start, width) in old_offsets.items():
            new_start, new_branch_width = new_offsets[key]
            if new_branch_width != width:
                raise RuntimeError(
                    f"branch {key} width changed: {width} -> {new_branch_width}"
                )
            new_head[:, new_start : new_start + width] = old_head[
                :, old_start : old_start + width
            ]
        for key in new_keys:
            new_start, width = new_offsets[key]
            new_head[:, new_start : new_start + width] = (
                torch.randn_like(new_head[:, new_start : new_start + width])
                * args.new_column_std
            )
        new_sd[head_key] = new_head
    new_sd[HEAD_BIAS] = old_sd[HEAD_BIAS].clone()
    new_model.policy.load_state_dict(new_sd)

    # Parity check: on identical old observations the new extractor should be
    # close to the old one when the local-map window is empty.
    old_obs = _random_observations(old_model.observation_space)
    new_obs = dict(old_obs)
    local_shape = new_model.observation_space["local_map"].shape
    new_obs["local_map"] = torch.zeros((1, *local_shape), dtype=torch.float32)
    with torch.no_grad():
        old_features = old_model.policy.extract_features(old_obs)
        new_features = new_model.policy.extract_features(new_obs)
    mean_abs_diff = float((old_features - new_features).abs().mean())
    mean_abs = float(old_features.abs().mean())
    relative = mean_abs_diff / max(mean_abs, 1e-12)

    new_model.num_timesteps = args.old_steps
    model_path = checkpoint_dir / f"model_{args.old_steps}.zip"
    new_model.save(str(model_path))

    if args.old_vecnormalize:
        old_config = _env_config(args.old_tier, args.rom, args.save_state)
        _remap_vecnormalize(
            args.old_vecnormalize,
            old_config,
            new_env,
            checkpoint_dir / "vecnormalize.pkl",
        )
    if args.old_first_seen:
        shutil.copy2(args.old_first_seen, out_dir / "first_seen.csv")

    metadata = {
        "old_checkpoint": str(Path(args.old_checkpoint).resolve()),
        "old_tier": args.old_tier,
        "new_tier": args.new_tier,
        "old_steps": args.old_steps,
        "old_components": list(old_model.observation_space.spaces),
        "new_components": list(new_model.observation_space.spaces),
        "old_head_width": int(old_width),
        "new_head_width": int(new_width),
        "new_branches": new_keys,
        "old_branch_offsets": old_offsets,
        "new_branch_offsets": new_offsets,
        "copied_tensors": copied,
        "skipped_tensors": skipped,
        "parity_mean_abs_diff": mean_abs_diff,
        "parity_relative_diff": relative,
    }
    (out_dir / "transfer.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))
    print(f"wrote {model_path}")
    new_env.close()


if __name__ == "__main__":
    main()
