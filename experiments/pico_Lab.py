import torchaudio
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

SR = 22050 # GriffinLim + Mel work best at 22k
N_MELS = 128
PATCH_SIZE = 16
N_FFT = 1024
HOP = 256

mel_transform = T.MelSpectrogram(sample_rate=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
db_transform = T.AmplitudeToDB()
griffin = T.GriffinLim(n_fft=N_FFT, hop_length=HOP)

def load_audio(path):
    """Loads mp3 or wav. Auto resamples to 22k and makes mono"""
    audio, sr = torchaudio.load(path) # works for mp3 and wav
    audio = audio.mean(0, keepdim=True) # stereo -> mono [1, T]
    if sr!= SR:
        audio = torchaudio.functional.resample(audio, sr, SR)
    return audio

def audio_to_patches(path_or_tensor):
    """
    INPUT: "song.mp3" OR tensor [1, T]
    OUTPUT: patches [N, 256], shape, mean, std
    """
    audio = path_or_tensor if isinstance(path_or_tensor, torch.Tensor) else load_audio(path_or_tensor)

    # 1. Audio -> Mel dB [1, 128, T]
    mel = mel_transform(audio)
    mel_db = db_transform(mel)

    # 2. Normalize for DiT
    mean, std = mel_db.mean(), mel_db.std()
    mel_norm = (mel_db - mean) / (std + 1e-5)

    # 3. Pad so H and W are divisible by PATCH_SIZE
    _, c, h, w = mel_norm.shape
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    mel_pad = F.pad(mel_norm, (0, pad_w, 0, pad_h))

    # 4. Cut into patches [N, 256]
    patches = mel_pad.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    patches = patches.contiguous().view(-1, PATCH_SIZE * PATCH_SIZE)

    return patches, (c, h+pad_h, w+pad_w), mean, std

def patches_to_audio(patches, shape, mean, std, original_len=None):
    """
    INPUT: patches [N, 256], shape, mean, std
    OUTPUT: audio tensor [1, T]
    """
    c, h, w = shape

    # 1. Glue patches back -> [1, 128, H, W]
    mel_pad = patches.view(h//PATCH_SIZE, w//PATCH_SIZE, c, PATCH_SIZE, PATCH_SIZE)
    mel_pad = mel_pad.permute(2,0,3,1,4).contiguous().view(c, h, w)

    # 2. Denormalize
    mel_db = mel_pad * std + mean

    # 3. dB -> Amplitude. NO DBToAmplitude error
    mel_amp = torch.pow(10.0, mel_db / 20.0)

    # 4. Mel -> Audio with GriffinLim
    audio = griffin(mel_amp)

    # 5. Crop back to original length if provided
    if original_len is not None:
        audio = audio[:, :original_len]

    return audio