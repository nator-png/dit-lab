from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .text import HashTokenizer


class ImageTextDataset(Dataset):
    """Image-caption dataset backed by a folder and a simple captions file.

    Each line in the caption file should be:
    `image_filename.ext caption text goes here`
    """

    def __init__(
        self,
        image_dir: str | Path,
        captions_file: str | Path,
        image_size: int = 64,
        vocab_size: int = 10_000,
        max_tokens: int = 77,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.samples = self._read_captions(Path(captions_file))
        self.tokenizer = HashTokenizer(vocab_size=vocab_size, max_length=max_tokens)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_name, caption = self.samples[index]
        image = Image.open(self.image_dir / image_name).convert("RGB")
        token_ids = self.tokenizer.encode(caption)
        return self.transform(image), token_ids

    @staticmethod
    def _read_captions(path: Path) -> list[tuple[str, str]]:
        samples: list[tuple[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    image_name, caption = stripped.split(maxsplit=1)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid caption line {line_number}: expected '<filename> <caption>'."
                    ) from exc
                samples.append((image_name, caption))
        return samples
