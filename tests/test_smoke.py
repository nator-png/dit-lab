import torch

from dit_lab.config import ImageDiTConfig, VideoDiTConfig
from dit_lab.models.image_dit import ImageDiT
from dit_lab.models.video_dit import VideoDiT


def test_image_dit_forward_shape() -> None:
    config = ImageDiTConfig(embedding_dim=32, layers=1, heads=4, image_size=16, patch_size=4)
    model = ImageDiT(config)
    output = model(
        torch.randn(2, 3, 16, 16),
        torch.randint(0, 1000, (2,)),
        torch.randn(2, 512),
    )
    assert output.shape == (2, 3, 16, 16)


def test_video_dit_forward_shape() -> None:
    config = VideoDiTConfig(
        embedding_dim=32,
        layers=1,
        heads=4,
        frames=2,
        image_size=16,
        patch_size=4,
    )
    model = VideoDiT(config)
    output = model(
        torch.randn(2, 3, 2, 16, 16),
        torch.randint(0, 1000, (2,)),
        torch.randn(2, 512),
    )
    assert output.shape == (2, 3, 2, 16, 16)
