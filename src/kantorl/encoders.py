"""Feature extractor that gives every observation component its own encoder.

Why this exists
---------------
The stock `MultiInputPolicy` extractor concatenates the raw vector components
into one MLP. Measured on 20M-step checkpoints, that network *ignores* five of
the seven components: randomising the 2560-bit event vector, badges, levels,
HP or action history changed the chosen action on **0%** of 250 transitions,
while randomising the screen changed it on 50-93%. The RAM observations were
dead weight, so the tier comparison was measuring the encoder rather than the
value of the observations.

CJBoey's improved Gen-1 baseline reports the same thing and solves it with a
modular architecture ("without this network architecture, the agent fails to
learn anything when given the same observation"). This is that idea in
miniature:

    screens        -> NatureCNN            -> 256 features
    each other key -> Linear -> LayerNorm -> Linear -> 64 features
    concatenation  -> Linear -> features_dim

Every branch gets its own normalisation, so a 2560-dimensional binary vector
cannot dominate or be drowned out by a 1-dimensional health ratio. The final
projection fixes the policy input width, so all tiers present the same
capacity to the policy regardless of how many components they observe.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from torch import nn


class TileMapEncoder(nn.Module):
    """Encode a small 2-D tile map with a learned per-tile embedding.

    Raw tile IDs are categorical, not ordinal: tile 0x3D is not "close to"
    tile 0x3E in any useful sense. A ``nn.Embedding`` lets the model learn
    which tiles behave alike (wall, floor, water, door), then a small CNN
    learns the local geometry. This avoids hand-authoring per-tileset
    walkability tables while still giving the policy a spatial view.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        out_dim: int = 64,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(256, embed_dim)
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * shape[0] * shape[1], out_dim),
            nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        indices = value.long().clamp(0, 255)
        embedded = self.embed(indices)  # (B, H, W, E)
        embedded = embedded.permute(0, 3, 1, 2).contiguous()
        return self.net(embedded)


class ModularExtractor(BaseFeaturesExtractor):
    """Encode each observation key separately, then fuse.

    Args:
        observation_space: Dict space produced by the environment.
        features_dim: Width of the fused feature vector handed to the policy.
        proj_dim: Width each vector component is projected to.
        cnn_features: Width of the image branch.

    Example:
        >>> kwargs = {"features_extractor_class": ModularExtractor,
        ...           "features_extractor_kwargs": {"features_dim": 512}}
        >>> model = PPO("MultiInputPolicy", env, policy_kwargs=kwargs)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 512,
        proj_dim: int = 64,
        cnn_features: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # SB3 2.9 does not keep the space on the extractor; we need nvec below.
        self._space = observation_space

        self.image_encoders = nn.ModuleDict()
        self.tile_encoders = nn.ModuleDict()
        self.vector_encoders = nn.ModuleDict()
        self.vector_kinds: dict[str, str] = {}

        fused = 0
        for key, space in observation_space.spaces.items():
            if isinstance(space, spaces.Box) and len(space.shape) == 3:
                self.image_encoders[key] = NatureCNN(space, features_dim=cnn_features)
                fused += cnn_features
                continue

            # Small 2-D tile maps (local_tiles / local_map) get an embedding
            # and a CNN. The global 48x48 map stays on the vector path so
            # existing checkpoints keep their architecture.
            if (
                isinstance(space, spaces.Box)
                and len(space.shape) == 2
                and space.dtype == np.uint8
                and max(space.shape) <= 21
            ):
                self.tile_encoders[key] = TileMapEncoder(space.shape, proj_dim)
                fused += proj_dim
                continue

            if isinstance(space, spaces.MultiDiscrete):
                in_dim = int(np.sum(space.nvec))
                kind = "discrete"
            elif isinstance(space, spaces.MultiBinary):
                in_dim = int(np.prod(space.shape))
                kind = "binary"
            else:
                in_dim = int(np.prod(space.shape))
                kind = "box"

            hidden = int(max(proj_dim, min(256, in_dim)))
            self.vector_encoders[key] = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, proj_dim),
                nn.ReLU(),
            )
            self.vector_kinds[key] = kind
            fused += proj_dim

        # Fixed-width fusion so every tier gives the policy the same input size.
        self.head = nn.Sequential(nn.Linear(fused, features_dim), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        for key, encoder in self.image_encoders.items():
            value = observations[key]
            if not torch.is_floating_point(value):
                # SB3 pre-scales image observations to [0, 1] before the
                # extractor runs; direct callers may hand over raw uint8.
                value = value.float() / 255.0
            parts.append(encoder(value))

        for key, encoder in self.tile_encoders.items():
            parts.append(encoder(observations[key]))

        for key, encoder in self.vector_encoders.items():
            value = observations[key].float()
            if self.vector_kinds[key] == "discrete":
                nvec = self._space[key].nvec
                one_hot = [
                    nn.functional.one_hot(
                        value[:, i].long(), num_classes=int(n)
                    ).float()
                    for i, n in enumerate(nvec)
                ]
                value = torch.cat(one_hot, dim=1)
            else:
                # 2-D components such as the 48x48 map need flattening.
                value = value.reshape(value.shape[0], -1)
            parts.append(encoder(value))

        return self.head(torch.cat(parts, dim=1))
