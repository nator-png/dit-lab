import torch
import torch.nn as nn
import numpy as np


config = {
          "patch_size":4,
          "embd":64,
          "heads":4,
          "head_dim":16,
          "qkv_bias":True,
          "layers":10,
          "steps":1000,
          "drop_rate":0.1,
          "cont_length":256,
          "num_channels":3,
          "image_size":64,
          "batch_size":1,
          "lr":0.01

          }




class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.patch_emb = nn.Conv2d(config["num_channels"],config["embd"],config["patch_size"],config["patch_size"])
        num_patches = int((config["image_size"]/config["patch_size"])**2)
        self.pos_emb = nn.Parameter(torch.rand(1,num_patches,config["embd"]))

    def forward(self,x):
        
        x_patch = self.patch_emb(x)
        x_patch = x_patch.flatten(2).transpose(1,2)
        x = x_patch + self.pos_emb
        
        return x




class TimestepEmb(nn.Module):
    def __init__(self,embd):
        super().__init__()

        self.ff = nn.Sequential(nn.Linear(embd,embd * 4),
                                nn.SiLU(),
                                nn.Linear(embd * 4,embd))
        
        self.emb = embd
    
    def forward(self,x):

        half_dim = self.emb//2

        freqs = torch.exp(torch.arange(half_dim,dtype=torch.float32) * -np.log(10000) / (half_dim-1))

        emb = x[:,None].float() * freqs[None,:]

        emb = torch.cat([emb.sin(),emb.cos()],dim=-1)

        return  self.ff(emb)
    
    


class DiTBlock(nn.Module):
    def __init__(self,embd):
        super().__init__()

        self.txt_att = nn.MultiheadAttention(embed_dim=embd,batch_first=True,num_heads=4)
        self.crs_att = nn.MultiheadAttention(embed_dim=embd,batch_first=True,num_heads=4)

        self.norm1 = nn.LayerNorm(embd)
        self.norm2 = nn.LayerNorm(embd)
        self.norm3 = nn.LayerNorm(embd)

        self.ff = nn.Sequential(nn.Linear(embd , embd * 4),
                                nn.GELU(),
                                nn.Linear(embd * 4,embd))
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),nn.Linear(embd,embd * 6))
        

    def forward(self,t_embd,text_emb,x):

        shift1,scale1,gate1,shift2,scale2,gate2 = self.adaLN_modulation(t_embd).chunk(6,dim = -1)

        norm_x = self.norm1(x) * (1 + scale1.unsqueeze(1)) + (shift1.unsqueeze(1))
        x = x + gate1.unsqueeze(1) * self.txt_att(norm_x,norm_x,norm_x)[0]

        norm_x = self.norm2(x) * (1 + scale2.unsqueeze(1)) + (shift2.unsqueeze(1))
        x = x + gate2.unsqueeze(1) * self.crs_att(norm_x,text_emb,text_emb)[0]

        x = self.ff(self.norm3(x))

        return x
    


class FullDiT(nn.Module):
    def __init__(self):
        super().__init__()

        self.patches = Embeddings()
        self.TimeStep_Emb = TimestepEmb(embd=64)
        self.text_Embedder = nn.Linear(512,64)
        self.D_block =  nn.ModuleList([DiTBlock(embd=64) for b in range(6)])
        self.final_layer = nn.Linear(64,(config["patch_size"]**2) * config["num_channels"])

        self.h_grid = config["image_size"]//config["patch_size"]
        self.w_grid = config["image_size"]//config["patch_size"]


    def forward(self,steps,text_emb,noisy):

        x = noisy
        x = self.patches(x)
        t_emb = self.TimeStep_Emb(steps)
        
        text_emb = self.text_Embedder(text_emb).unsqueeze(1)
        
        for block in self.D_block:

            x = block(t_emb,text_emb,x)

        
        
        x = self.final_layer(x)

        b,_,_ = x.shape
        x = x.view(b,self.h_grid,self.w_grid,3,config["patch_size"],config["patch_size"]).permute(0 ,3 ,1 ,4 ,2 ,5).contiguous()

        return x
    




torch.manual_seed(123)
steps = torch.randint(0, config["steps"],(config["batch_size"],))
text_emb = torch.randn(1,512)
noisy = torch.randn(1,3,64,64)

model = FullDiT()


print(model(steps = steps,text_emb = text_emb,noisy = noisy))







        


        





"""

import torch
import torch.nn as nn
import numpy as np

config = {
    "patch_size":4,
    "embd":64,
    "heads":4,
    "layers":6,
    "steps":1000,
    "num_channels":3,
    "image_size":64,
    "batch_size":1,
}

class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_emb = nn.Conv2d(config["num_channels"],config["embd"],config["patch_size"],config["patch_size"])
        num_patches = (config["image_size"]//config["patch_size"])**2
        self.pos_emb = nn.Parameter(torch.rand(1,num_patches,config["embd"]))

    def forward(self,x):
        x_patch = self.patch_emb(x) # [B, embd, H', W']
        x_patch = x_patch.flatten(2).transpose(1,2) # [B, N, embd]
        x = x_patch + self.pos_emb
        return x

class TimestepEmb(nn.Module):
    def __init__(self,embd):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(embd,embd * 4), nn.SiLU(), nn.Linear(embd * 4,embd))
        self.emb = embd

    def forward(self,x):
        half_dim = self.emb//2
        device = x.device
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -np.log(10000) / (half_dim-1))
        emb = x[:,None].float() * freqs[None,:]
        emb = torch.cat([emb.sin(),emb.cos()],dim=-1)
        return self.ff(emb)

class DiTBlock(nn.Module):
    def __init__(self,embd, heads=4):
        super().__init__()
        self.txt_att = nn.MultiheadAttention(embed_dim=embd, batch_first=True, num_heads=heads)
        self.crs_att = nn.MultiheadAttention(embed_dim=embd, batch_first=True, num_heads=heads)
        self.norm1 = nn.LayerNorm(embd)
        self.norm2 = nn.LayerNorm(embd)
        self.norm3 = nn.LayerNorm(embd)
        self.ff = nn.Sequential(nn.Linear(embd, embd * 4), nn.GELU(), nn.Linear(embd * 4,embd))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),nn.Linear(embd,embd * 6))

    def forward(self,t_embd,text_emb,x):
        shift1,scale1,gate1,shift2,scale2,gate2 = self.adaLN_modulation(t_embd).chunk(6,dim = -1)
        norm_x = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + gate1.unsqueeze(1) * self.txt_att(norm_x,norm_x,norm_x)[0]
        norm_x = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.crs_att(norm_x,text_emb,text_emb)[0]
        x = x + self.ff(self.norm3(x)) # residual
        return x

class FullDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patches = Embeddings()
        self.TimeStep_Emb = TimestepEmb(embd=64)
        self.text_Embedder = nn.Linear(512,64)
        self.D_block = nn.ModuleList([DiTBlock(embd=64, heads=4) for _ in range(config["layers"])])
        self.final_layer = nn.Linear(64,(config["patch_size"]**2) * config["num_channels"])
        self.h = self.w = config["image_size"] // config["patch_size"]

    def forward(self,steps,text_emb,noisy):
        x = self.patches(noisy)
        t_emb = self.TimeStep_Emb(steps)
        text_emb = self.text_Embedder(text_emb).unsqueeze(1)
        for block in self.D_block:
            x = block(t_emb,text_emb,x)
        x = self.final_layer(x)
        b = x.shape[0]
        x = x.view(b, self.h, self.w, config["num_channels"], config["patch_size"], config["patch_size"])
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, config["num_channels"], config["image_size"], config["image_size"])
        return x

torch.manual_seed(123)
steps = torch.randint(0, config["steps"],(config["batch_size"],))
text_emb = torch.randn(config["batch_size"],512)
noisy = torch.randn(config["batch_size"],3,64,64)

model = FullDiT()
print(model(steps=steps, text_emb=text_emb, noisy=noisy).shape) # [1, 3, 64, 64]

"""