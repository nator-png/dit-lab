from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Training options shared by the example trainers."""

    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    timesteps: int = 1000
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 0
    save_every: int = 1


@dataclass(frozen=True)
class ImageDiTConfig:
    """Model dimensions for the image Diffusion Transformer."""

    embedding_dim: int = 256
    patch_size: int = 4
    layers: int = 12
    heads: int = 4
    image_size: int = 64
    channels: int = 3
    text_dim: int = 512


@dataclass(frozen=True)
class VideoDiTConfig:
    """Model dimensions for the video Diffusion Transformer."""

    embedding_dim: int = 256
    patch_size: int = 4
    layers: int = 12
    heads: int = 4
    frames: int = 16
    image_size: int = 64
    channels: int = 3
    text_dim: int = 512
