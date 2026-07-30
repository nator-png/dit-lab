from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms


class AudioPatchProcessor:
    """Convert audio waveforms to normalized mel patches and back."""

    def __init__(
        self,
        sample_rate: int = 22_050,
        n_fft: int = 1024,
        n_mels: int = 128,
        hop_length: int = 256,
        patch_size: int = 16,
    ) -> None:
        self.sample_rate = sample_rate
        self.patch_size = patch_size
        self.mel = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            power=2.0,
        )
        self.to_db = transforms.AmplitudeToDB(stype="power")
        self.griffin_lim = transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length)

    def load_audio(self, path: str | Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform

    def audio_to_patches(
        self,
        audio_or_path: torch.Tensor | str | Path,
    ) -> tuple[torch.Tensor, tuple[int, int, int], torch.Tensor, torch.Tensor]:
        waveform = (
            audio_or_path
            if isinstance(audio_or_path, torch.Tensor)
            else self.load_audio(audio_or_path)
        )
        mel_db = self.to_db(self.mel(waveform))
        mean, std = mel_db.mean(), mel_db.std()
        normalized = (mel_db - mean) / (std + 1e-5)

        channels, height, width = normalized.shape
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        padded = F.pad(normalized, (0, pad_w, 0, pad_h))

        patches = padded.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(-1, self.patch_size * self.patch_size)
        return patches, (channels, height + pad_h, width + pad_w), mean, std
