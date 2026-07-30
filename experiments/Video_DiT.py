"""
VIDEO DiT - Text to Video 16 frames 64x64*
VIDEO DiT - Same as Image but 3D patches over Time+Space
Input: [B, 3, 16, 64, 64] -> Output: [B, 3, 16, 64, 64] noise prediction
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

config_vid = {"embd":256, "patch_size":4, "layers":12, "heads":4, "frames":16, "image_size":64,
              "num_channels":3, "head_dim":64, "batch_size":8, "lr":1e-4, "timesteps":1000}

class TimestepEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
    def forward(self, t):
        half = config_vid["embd"]//2
        freq = torch.exp(torch.arange(half, dtype=torch.float32) * -np.log(10000)/(half-1))
        emb = t[:,None].float() * freq[None,:].to(t.device)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)

class VideoDiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim))

    def forward(self, x, t_emb, text_emb):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(t_emb).chunk(6, dim=1)
        x_norm = self.norm1(x) * (1+sc1.unsqueeze(1)) + s1.unsqueeze(1)
        x = x + g1.unsqueeze(1) * self.attn(x_norm, x_norm, x_norm)[0]
        x = x + g2.unsqueeze(1) * self.cross_attn(self.norm2(x), text_emb, text_emb)[0]
        x = x + self.ff(self.norm3(x))
        return x

class VideoDiT(nn.Module):
    def __init__(self):
        super().__init__()
        T, H, P = config_vid["frames"], config_vid["image_size"], config_vid["patch_size"]
        num_patches = T * (H//P) * (H//P) # 16 * 16 * 16 = 4096 patches
        patch_dim = P * P * config_vid["num_channels"] # 4*4*3=48

        # Conv3d: patches over time dimension too. kernel=(1,P) = patch spatial only, keep time
        self.patch_embed = nn.Conv3d(3, config_vid["embd"], kernel_size=(1,P), stride=(1,P))

        # Positional embedding for spatio-temporal patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, config_vid["embd"]))
        self.text_proj = nn.Linear(512, config_vid["embd"])
        self.timestep_embed = TimestepEmbed(config_vid["embd"])
        self.blocks = nn.ModuleList([VideoDiTBlock(config_vid["embd"], config_vid["heads"]) for _ in range(config_vid["layers"])])
        self.final_layer = nn.Linear(config_vid["embd"], patch_dim)

    def forward(self, x_noisy, timesteps, text_emb):
        # x_noisy: [B, 3, 16, 64, 64] video
        x = self.patch_embed(x_noisy) # [B, 256, 16, 16, 16]
        x = x.flatten(2).transpose(1, 2) + self.pos_embed # [B, 4096, 256]
        t_emb = self.timestep_embed(timesteps)
        text_emb = self.text_proj(text_emb).unsqueeze(1)
        for blk in self.blocks: x = blk(x, t_emb, text_emb)
        x = self.final_layer(x) # [B, 4096, 48]

        # Unpatchify back to video
        B = x.shape[0]; P=config_vid["patch_size"]; H=config_vid["image_size"]; T=config_vid["frames"]
        x = x.view(B, T, H//P, H//P, 3, P, P).permute(0,4,1,2,5,3,6).contiguous()
        return x.view(B, 3, T, H, H)

#Training + Inference same as Image but batch_size=8 due to memory
