from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .diffusion import add_noise
from .text import MeanTextEncoder


def train_noise_predictor(
    model: torch.nn.Module,
    dataloader: DataLoader,
    text_encoder: MeanTextEncoder,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    """Standard training loop for DiT-style noise prediction."""

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    text_encoder.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(text_encoder.parameters()),
        lr=config.learning_rate,
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        text_encoder.train()
        running_loss = 0.0

        for clean_batch, token_ids in dataloader:
            clean_batch = clean_batch.to(device)
            token_ids = token_ids.to(device)
            timesteps = torch.randint(
                0,
                config.timesteps,
                (clean_batch.shape[0],),
                device=device,
            )

            noisy_batch, noise_target = add_noise(clean_batch, timesteps, config.timesteps)
            text_embedding = text_encoder(token_ids)
            predicted_noise = model(noisy_batch, timesteps, text_embedding)
            loss = F.mse_loss(predicted_noise, noise_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / max(len(dataloader), 1)
        print(f"epoch={epoch} loss={average_loss:.4f}")

        if epoch % config.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "epoch": epoch,
                    "loss": average_loss,
                },
                checkpoint_dir / f"dit_epoch_{epoch:04d}.pt",
            )
