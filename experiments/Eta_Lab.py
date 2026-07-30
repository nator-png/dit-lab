import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path


pth = Path("C:/Users/block/Desktop/AI Universe/AI Datasets/Test Songs")
file = Path.glob(pth,"*mp3")

file_list = []

for f in file:
    file_list.append(f)


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

    # if any sample was padded, stack masks. else None
    if masks[0] is not None:
        masks = torch.stack(masks) # [B, 430]
        return mels, masks
    return mels, None

loader = DataLoader(
    AudioChunkDataset(file_list),
    batch_size=10,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)





for x,y in loader:
    print(x.shape)