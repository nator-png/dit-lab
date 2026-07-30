import hashlib

import torch
from torch import nn


class HashTokenizer:
    """Tiny deterministic tokenizer for experiments without external text models."""

    def __init__(self, vocab_size: int = 10_000, max_length: int = 77) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length

    def encode(self, text: str) -> torch.Tensor:
        words = text.lower().split()[: self.max_length]
        token_ids = [
            int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.vocab_size
            for word in words
        ]
        token_ids += [0] * (self.max_length - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)


class MeanTextEncoder(nn.Module):
    """Embedding-table text encoder used as a lightweight CLIP placeholder."""

    def __init__(
        self,
        vocab_size: int = 10_000,
        text_dim: int = 512,
        max_length: int = 77,
    ) -> None:
        super().__init__()
        self.tokenizer = HashTokenizer(vocab_size=vocab_size, max_length=max_length)
        self.embedding = nn.Embedding(vocab_size, text_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids).mean(dim=1)

    def encode_texts(self, captions: list[str], device: torch.device) -> torch.Tensor:
        token_ids = torch.stack([self.tokenizer.encode(caption) for caption in captions]).to(device)
        return self.forward(token_ids)
