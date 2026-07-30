import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dit_lab.config import ImageDiTConfig, TrainingConfig
from dit_lab.datasets import ImageTextDataset
from dit_lab.models.image_dit import ImageDiT
from dit_lab.text import MeanTextEncoder
from dit_lab.train import train_noise_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the image DiT noise predictor.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ImageDiTConfig(image_size=args.image_size)
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=str(args.checkpoint_dir),
    )
    dataset = ImageTextDataset(args.image_dir, args.captions, image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    model = ImageDiT(model_config)
    text_encoder = MeanTextEncoder(text_dim=model_config.text_dim)
    train_noise_predictor(model, dataloader, text_encoder, train_config, device)


if __name__ == "__main__":
    main()
