import torch
import torch.nn.functional as Fn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


sample_rate = 22050
n_fft = 1024
n_mels = 128
hop = 256
patch_size = 16

mel_spec = T.MelSpectrogram(sample_rate=sample_rate,n_fft=n_fft,n_mels=n_mels,hop_length=hop,power=2.0)
mel_db = T.AmplitudeToDB(stype="power",top_db=None)

mel_basis = T.MelScale(n_stft= n_fft//2+1,n_mels=n_mels,sample_rate=sample_rate).fb
mel_basis_inverse = torch.linalg.pinv(mel_basis)
grifin = T.GriffinLim(n_fft=n_fft,n_iter=64,hop_length=hop,power=2.0)




def load_aud(pth):

    audio_wave_tensor,sr = torchaudio.load(pth)

    if sr != sample_rate:
        audio_wave_tensor = F.resample(audio_wave_tensor,sr,sample_rate)
    
    audio_wave_tensor = audio_wave_tensor.mean(0,keepdim=True)

    return audio_wave_tensor

def audio_to_patches(audio_tensor):
    aud_spec = mel_spec(audio_tensor)
    c,h,w = aud_spec.shape
    aud_db = mel_db(aud_spec)
    

    mean = aud_db.mean()
    std = aud_db.std()
    aud_norm = (aud_db - mean)/(std + 1e-5)

    h_pad = (patch_size - h % patch_size) % patch_size
    w_pad = (patch_size - w % patch_size) % patch_size

    aud_pad = Fn.pad(aud_norm,(0,w_pad,0,h_pad)).unsqueeze(0)
    num_patches = aud_pad.unfold(2,patch_size,patch_size).unfold(3,patch_size,patch_size)
    #num_patches = num_patches.permute(0,1,2,3,4,5).reshape(-1,patch_size * patch_size)

    return num_patches,(c,h + h_pad,w + w_pad),mean,std

def patches_to_audio(patches,shape,mean,std,original_len=None):
    c,h,w = shape

    h_grid = h // patch_size
    w_grid = w // patch_size

    aud_spec = patches.reshape(1,c,h_grid,w_grid,patch_size,patch_size)
    aud_spec = aud_spec.permute(0,1,2,4,3,5).reshape(1,c,h,w).squeeze(0)
    aud_denorm = aud_spec * std + mean 
    aud_pow = torch.pow(10.0,aud_denorm/10.0).squeeze(0)
    aud_basis = mel_basis_inverse.T @ aud_pow
    aud_basis = torch.clamp(aud_basis,min = 1e-10)
    output = grifin(aud_basis.unsqueeze(0))
    
    

    return output#[:,:original_len]





if __name__ == "__main__":
    print("nano_Lab is an archived audio patching experiment. Call load_aud/audio_to_patches manually.")




