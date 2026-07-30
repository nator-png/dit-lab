import math

import torch
from torch import nn


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a small MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * -math.log(10_000)
            / max(half_dim - 1, 1)
        )
        embedding = timesteps[:, None].float() * frequencies[None, :]
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=-1)
        return self.mlp(embedding)


class DiTBlock(nn.Module):
    """Transformer block with adaLN timestep conditioning and text cross-attention."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def forward(
        self,
        patches: torch.Tensor,
        timestep_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_ln(
            timestep_embedding
        ).chunk(6, dim=1)

        normalized = self.norm1(patches) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        patches = patches + gate1.unsqueeze(1) * self.self_attention(
            normalized, normalized, normalized
        )[0]

        normalized = self.norm2(patches) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        patches = patches + gate2.unsqueeze(1) * self.cross_attention(
            normalized, text_embedding, text_embedding
        )[0]

        return patches + self.feed_forward(self.norm3(patches))
