import torch
from torch import nn

from dit_lab.config import ImageDiTConfig
from dit_lab.models.common import DiTBlock, TimestepEmbedding


class ImageDiT(nn.Module):
    """Minimal text-conditioned Diffusion Transformer for square RGB images."""

    def __init__(self, config: ImageDiTConfig | None = None) -> None:
        super().__init__()
        self.config = config or ImageDiTConfig()
        cfg = self.config
        patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels
        patches_per_side = cfg.image_size // cfg.patch_size
        num_patches = patches_per_side * patches_per_side

        self.patch_embed = nn.Conv2d(
            cfg.channels,
            cfg.embedding_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
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
        noisy_image: torch.Tensor,
        timesteps: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        patches = self.patch_embed(noisy_image).flatten(2).transpose(1, 2)
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
            grid,
            grid,
            cfg.channels,
            cfg.patch_size,
            cfg.patch_size,
        )
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(batch, cfg.channels, cfg.image_size, cfg.image_size)
