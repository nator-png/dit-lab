import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import hashlib
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os

try:
    import sounddevice as sd
except ImportError:
    sd = None


config = {
    'embd':64,
    'layers':4,
    'heads':8,
    'text_embd':64,
    'batch_size':1,
    'patch_size':16,
    'channels':1,
    'n_mels':128,
    'n_fft':1024,
    'hop_length':256,
    'sample_rate':22050,
    'power':2.0,
    'stype':'power',
    'vocab_size':100000,
    'lr':3e-4,
    'steps':1000,
    'max_tok':50,
    'epochs':5
    



         }


mel_spectrogram = T.MelSpectrogram(sample_rate=config['sample_rate'],n_fft=config['n_fft'],n_mels=config['n_mels'],hop_length=config['hop_length'],power=config['power'])
mel_DB = T.AmplitudeToDB(stype=config['stype'])
mel_basis = T.MelScale(n_mels=config['n_mels'],sample_rate=config['sample_rate'],n_stft=config['n_fft']//2+1).fb
mel_basis_inverse = torch.linalg.pinv(mel_basis)
griffin = T.GriffinLim(n_fft=config['n_fft'],hop_length=config['hop_length'],power=config['power'])



class Embeddings(nn.Module):
    
    def __init__(self,c,h,w,embd = config['embd'],patch_size = config['patch_size'],batch_size = config['batch_size']):
        super().__init__()

        num_patches = (h//patch_size) * (w//patch_size)
        self.projections = nn.Conv2d(c,embd,patch_size,patch_size)
        self.pos_encoding = nn.Parameter(torch.rand(1,num_patches,embd))

    def forward(self,x):

        aud_img = self.projections(x)
        b = aud_img.shape[0]
        aud_img = aud_img.permute(0,2,3,1).reshape(b,-1,config['embd'])
        return aud_img + self.pos_encoding
    


class TimestepEmbed(nn.Module):

    def __init__(self,embd = config['embd']):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embd,embd * 4),
            nn.SiLU(),
            nn.Linear(embd * 4,embd)
        )
        self.half_dim = embd//2

    def forward(self,t):

        freqs = torch.exp(torch.arange(self.half_dim, device=t.device) * -np.log(10000)/self.half_dim - 1)
        embd = t[:,None] * freqs[None,:]
        embd = torch.cat([embd.sin(),embd.cos()],dim=-1)

        return self.mlp(embd)
    


class DiTBlock(nn.Module):
    def __init__(self,embd = config['embd'],num_heads = config['heads'],text_emb = config['text_embd']):
        super().__init__()

        self.text_cross_att = nn.MultiheadAttention(embed_dim=embd,num_heads=num_heads,batch_first=True)
        self.aud_img_att = nn.MultiheadAttention(embed_dim=embd,num_heads=num_heads,batch_first=True)
        self.norm1 = nn.LayerNorm(embd)
        self.norm2 = nn.LayerNorm(embd)
        self.norm3 = nn.LayerNorm(embd)
        self.txt_emb = nn.Linear(text_emb,embd)
        self.adaLN_Modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embd,embd * 6)

        )
        self.ff = nn.Sequential(
            nn.Linear(embd,embd * 4),
            nn.GELU(),
            nn.Linear(embd * 4,embd)
        )
        

    def forward(self,aud_img,text_emb,t_emb):

        scale1,shift1,gate1,scale2,shift2,gate2 = self.adaLN_Modulation(t_emb).chunk(6,-1)

        text_emb = self.txt_emb(text_emb).unsqueeze(1)

        x = self.norm1(aud_img) * (scale1.unsqueeze(0) + 1) + shift1.unsqueeze(0)
        x = x + gate1.unsqueeze(1) * self.aud_img_att(aud_img,aud_img,aud_img)[0]

        x = self.norm2(x) * (scale2.unsqueeze(0) + 1) + shift2.unsqueeze(0)
        x = x + gate2.unsqueeze(0) * self.text_cross_att(x,text_emb,text_emb)[0]

        return self.ff(self.norm3(x))
    




class DiTModel(nn.Module):
    def __init__(self,c,h,w,embd = config['embd'],layers = config['layers'],channels = config['channels'],patch_size = config['patch_size']):
        super().__init__()

        self.Embeddings = Embeddings(c,h,w)
        self.TimeStepEmb = TimestepEmbed()
        self.DiTBlock = nn.ModuleList([DiTBlock() for _ in range(layers)])
        self.num_patches = (h//patch_size) * (w//patch_size)
        patches = c*patch_size*patch_size
        self.final_layer = nn.Linear(embd,patches)
        self.h_grid = h//patch_size
        self.w_grid = w//patch_size
        self.patch_size = patch_size
        self.c = c
        self.h = h
        self.w = w


    def forward(self,aud_img,text_emb,t):
        t_emb = self.TimeStepEmb(t)
        aud_img = self.Embeddings(aud_img)

        for block in self.DiTBlock:
            aud_img = block(aud_img,text_emb,t_emb)
        x = aud_img
        x = self.final_layer(x)
        b = x.shape[0]
        x = x.reshape(b,self.h_grid,self.w_grid,self.c,self.patch_size,self.patch_size).permute(0,3,1,4,2,5)
        x = x.reshape(b,self.c,self.h,self.w)
        
        return x
    

"""
aud_img = torch.rand(1,1,128,14000)
_,c,h,w = aud_img.shape
t = torch.tensor([50])
text_emb = torch.rand(1,1,64).mean(0)

model = DiTModel(c=c,h=h,w=w)

print(model(aud_img,text_emb,t).shape)


"""
def aud_to_patches(pth,patch_size = config['patch_size']):

    audio_wave_tensor,sr = torchaudio.load(pth)

    if sr != config['sample_rate']:
        audio_wave_tensor = F.resample(audio_wave_tensor,sr,config['sample_rate'])
    audio_wave_tensor = audio_wave_tensor.mean(0,keepdim=True)

    audio_img = mel_spectrogram(audio_wave_tensor)
    aud_img_db = mel_DB(audio_img)
    mean,std = aud_img_db.mean(),aud_img_db.std()
    aud_img_db = (aud_img_db-mean)/std
    _,h,w = aud_img_db.shape
    h_pad = (patch_size - h % patch_size) % patch_size
    w_pad = (patch_size - w % patch_size) % patch_size
    aud_img_db = Fn.pad(aud_img_db,(0,w_pad,0,h_pad))

    return aud_img_db


class Tokenizer:
    def __init__(self,vocab_size = config['vocab_size'],max_tok = config['max_tok'] ):

        self.vocab_size = vocab_size
        self.max_tok = max_tok




    def encode(self,text):
        
       
        text_ids = [int(hashlib.md5(t.encode()).hexdigest(),16) % self.vocab_size for t in text.split()[:self.max_tok]]
        text_ids += [0] * (self.max_tok - len(text_ids))
        text_ids = torch.tensor(text_ids)

        

        return text_ids
        







class AudioText(Dataset):
    def __init__(self,audio_path = "data/audio",captions = "data/audio_captions.txt"):
        

        self.audio_path = audio_path
        self.captions = captions
        self.data = []


        with open(captions,"r") as f:
            
            for line in f:
                aud_name,caption = line.strip().split(" ",1)
                self.data.append((aud_name,caption))
            
        
        self.Tokenizer = Tokenizer()
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self,index):

        aud_name,caption = self.data[index]

        text_ids = self.Tokenizer.encode(caption)
        text_emb = nn.Embedding(config['vocab_size'],config['text_embd'])(text_ids).mean(0)

        aud_loc = os.path.join(self.audio_path,aud_name)
        aud_img = aud_to_patches(aud_loc)

        return aud_img,text_emb
    
def train_func(model,dataloader,epochs = config['epochs'],lr = config['lr'],t_steps = config['steps']):
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

    for epoch in range(epochs):
        
        loss = torch.zeros(0)

        for aud_clip,text_prompt in dataloader:
            optimizer.zero_grad()
            noisy_aud = torch.randn_like(aud_clip)
            b,_,_,_ = noisy_aud.shape
            t_step = torch.randint(0,t_steps,(b,))
            alpha = t_step/t_steps
            prediction = model(noisy_aud,text_prompt,t_step)

            prediction = (prediction + noisy_aud) * alpha
            prediction_loss = Fn.mse_loss(prediction,aud_clip)
            prediction_loss.backward()
            optimizer.step()

            loss += prediction_loss
        
        print(f"Current_loss: {loss.item():.4f} ")

        loss = torch.zeros(0)



#train_func()

            

def Generate_aud(prompt,model,t_steps = config['steps'],steps = 50,play_audio=False):

    aud = torch.randn(1,1,128,256)  
    text_ids = Tokenizer().encode(prompt)
    text_emb = nn.Embedding(config['vocab_size'],config['text_embd'])(text_ids).mean(0).unsqueeze(0)

    step_size = t_steps//steps

    for t in reversed(range(1,t_steps,step_size)):
        t_embed = torch.tensor([t])
        prediction = Model(aud,text_emb,t_embed)
        aud = (aud - prediction) * 0.1

        
    aud = torch.pow(10.0,aud/10.0).squeeze(0).squeeze(0)
    aud = (mel_basis_inverse.T @ aud).clamp(min=1e-10)
    aud = griffin(aud).detach().numpy().T
    if play_audio:
        if sd is None:
            raise RuntimeError("Install sounddevice to play generated audio.")
        sd.play(aud,samplerate=config['sample_rate'])
        sd.wait()
    


    return aud



if __name__ == "__main__":
    print("Audio_DiT is an archived experiment. Use src/dit_lab and scripts/ for training.")
