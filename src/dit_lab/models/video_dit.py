import torch
from torch import nn

from dit_lab.config import VideoDiTConfig
from dit_lab.models.common import DiTBlock, TimestepEmbedding


class VideoDiT(nn.Module):
    """Minimal text-conditioned DiT for short clips shaped [B, C, T, H, W]."""

    def __init__(self, config: VideoDiTConfig | None = None) -> None:
        super().__init__()
        self.config = config or VideoDiTConfig()
        cfg = self.config
        patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels
        grid = cfg.image_size // cfg.patch_size
        num_patches = cfg.frames * grid * grid

        self.patch_embed = nn.Conv3d(
            cfg.channels,
            cfg.embedding_dim,
            kernel_size=(1, cfg.patch_size, cfg.patch_size),
            stride=(1, cfg.patch_size, cfg.patch_size),
        )
        self.position_embed = nn.Parameter(torch.randn(1, num_patches, cfg.embedding_dim))
        self.text_projection = nn.Linear(cfg.text_dim, cfg.embedding_dim)
        self.timestep_embed = TimestepEmbedding(cfg.embedding_dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(cfg.embedding_dim, cfg.heads) for _ in range(cfg.layers)]
        )
        self.final_layer = nn.Linear(cfg.embedding_dim, patch_dim)

    def forward(
        self,
        noisy_video: torch.Tensor,
        timesteps: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        patches = self.patch_embed(noisy_video).flatten(2).transpose(1, 2)
        patches = patches + self.position_embed

        timestep_embedding = self.timestep_embed(timesteps)
        text_embedding = self.text_projection(text_embedding).unsqueeze(1)
        for block in self.blocks:
            patches = block(patches, timestep_embedding, text_embedding)

        patches = self.final_layer(patches)
        return self._unpatchify(patches)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch = patches.shape[0]
        grid = cfg.image_size // cfg.patch_size
        patches = patches.view(
            batch,
            cfg.frames,
            grid,
            grid,
            cfg.channels,
            cfg.patch_size,
            cfg.patch_size,
        )
        patches = patches.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
        return patches.view(batch, cfg.channels, cfg.frames, cfg.image_size, cfg.image_size)
