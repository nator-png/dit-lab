import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path


FRAMES_10S = 430

class AudioChunkDataset(Dataset):
    def __init__(self, mel_files):
        self.mel_files = mel_files # list of paths to [128][T] tensors

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel,sr = torchaudio.load(self.mel_files[idx]) # [128, T]
        T = mel.shape[1]

        # RULE 1: Song shorter than 10s -> PAD
        if T < FRAMES_10S:
            pad_len = FRAMES_10S - T
            mel = torch.nn.functional.pad(mel, (0, pad_len))
            mask = torch.cat([torch.ones(T), torch.zeros(pad_len)]) # 1=real, 0=pad
            return mel, mask

        # RULE 2: Song longer than 10s -> RANDOM CROP
        start = random.randint(0, T - FRAMES_10S)
        mel = mel[:, start:start+FRAMES_10S]
        return mel, None # no mask needed

def collate_fn(batch):
    mels, masks = zip(*batch)
    mels = torch.stack(mels) # [B, 128, 430]

    if any(mask is not None for mask in masks):
        fixed_masks = [
            mask if mask is not None else torch.ones(FRAMES_10S)
            for mask in masks
        ]
        return mels, torch.stack(fixed_masks)
    return mels, None


def build_loader(audio_dir="data/audio", batch_size=10):
    file_list = list(Path(audio_dir).glob("*.mp3"))
    return DataLoader(
        AudioChunkDataset(file_list),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    for x, _ in build_loader():
        print(x.shape)
