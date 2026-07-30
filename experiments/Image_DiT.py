"""
IMAGE DiT - Diffusion Transformer for Text-to-Image
From scratch. No external DiT libs. Only PyTorch fundamentals.
Training: Denoising Diffusion + Text Conditioning
Inference: DDIM sampling from random noise
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os



#==================== CONFIG ====================
config = {
    "embd": 256, # Embedding dimension - Transformer width
    "patch_size": 4, # 4x4 patches. 64/4 = 16 patches per side
    "layers": 12, # 12 Transformer blocks like ViT-Base
    "heads": 4, # 4 attention heads
    "image_size": 64, # 64x64 images for fast training
    "num_channels": 3, # RGB
    "head_dim": 64, # 256/4 = 64 per head
    "drop_rate": 0.1,
    "batch_size": 32,
    "lr": 1e-4,
    "timesteps": 1000 # Diffusion steps
}

#==================== 1. TIMESTEP EMBEDDING ====================
#Diffusion needs to know "what step am I at". Step 999 = very noisy, Step 1 = almost clean
class TimestepEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Sinusoidal + MLP. Same idea as Transformer positional encoding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(), # SiLU = smooth activation, better than ReLU for diffusion
            nn.Linear(dim*4, dim)
        )


    def forward(self, t):
        """
        t: [B] tensor of timesteps like [999, 500, 1]
        Returns: [B, dim] embedding vector
        """
        half_dim = config["embd"] // 2
        # Create frequency bands: 1, 10, 100, 1000...
        freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -np.log(10000) / (half_dim - 1))
        # t * freq = creates sine/cosine waves at different frequencies
        emb = t[:, None].float() * freq[None, :].to(t.device)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1) # [B, dim]
        return self.mlp(emb)

#==================== 2. DiT BLOCK WITH ADA-LN + CROSS ATTENTION ====================
class DiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True) # For text conditioning
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(), # GELU = what GPT/Transformer uses
            nn.Linear(dim*4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # adaLN = Adaptive Layer Norm. Timestep controls scale/shift/gate
        # This is the "DiT" innovation from "Scalable Diffusion Models with Transformers"
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim) # 6 outputs: shift1, scale1, gate1, shift2, scale2, gate2
        )

    def forward(self, x, t_emb, text_emb):
        """
        x: [B, N, dim] - image patches
        t_emb: [B, dim] - timestep embedding
        text_emb: [B, 1, dim] - text embedding from text encoder
        """
        # Get 6 modulation parameters from timestep
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(t_emb).chunk(6, dim=1)

        # Block 1: Self Attention with adaLN
        # x = x + gate * Attention( LayerNorm(x) * scale + shift )
        x_norm = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + gate1.unsqueeze(1) * self.attn(x_norm, x_norm, x_norm)[0]

        # Block 2: Cross Attention - text attends to image
        x_norm = self.norm2(x)
        x = x + gate2.unsqueeze(1) * self.cross_attn(x_norm, text_emb, text_emb)[0]

        # Block 3: Feed Forward
        x = x + self.ff(self.norm3(x))
        return x

#==================== 3. FULL IMAGE DiT MODEL ====================
class ImageDiT(nn.Module):
    def __init__(self):
        super().__init__()
        P = config["patch_size"]
        H = config["image_size"]
        num_patches = (H // P) ** 2 # 64/4 = 16, so 16*16 = 256 patches
        patch_dim = P * P * config["num_channels"] # 4*4*3 = 48 values per patch

        # Convert image to patches: Conv2d with stride=kernel_size does this
        self.patch_embed = nn.Conv2d(config["num_channels"], config["embd"], kernel_size=P, stride=P)

        # Learnable positional embedding - tells model "this patch is top-left"
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, config["embd"]))

        # Project text to same dimension as image patches
        self.text_proj = nn.Linear(512, config["embd"])

        # Timestep encoder
        self.timestep_embed = TimestepEmbed(config["embd"])

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(config["embd"], config["heads"]) for _ in range(config["layers"])
        ])

        # Final layer: predict noise for each patch pixel
        self.final_layer = nn.Linear(config["embd"], patch_dim)
        self.patch_dim = patch_dim

    def forward(self, x_noisy, timesteps, text_emb):
        """
        x_noisy: [B, 3, 64, 64] - noisy image
        timesteps: [B] - diffusion step
        text_emb: [B, 512] - text embedding
        Returns: [B, 3, 64, 64] - predicted noise
        """
        # 1. Patchify image
        x = self.patch_embed(x_noisy) # [B, 256, 16, 16]
        x = x.flatten(2).transpose(1, 2) # [B, 256 patches, 256 dim]
        x = x + self.pos_embed # Add position info

        # 2. Get embeddings
        t_emb = self.timestep_embed(timesteps) # [B, 256]
        text_emb = self.text_proj(text_emb).unsqueeze(1) # [B, 1, 256]

        # 3. Pass through Transformer blocks
        for blk in self.blocks:
            x = blk(x, t_emb, text_emb)

        # 4. Predict noise for each patch
        x = self.final_layer(x) # [B, 256, 48]

        # 5. Unpatchify back to image
        B = x.shape[0]
        x = x.view(B, 16, 16, 3, 4, 4).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 3, 64, 64)
        return x

#==================== 4. DATASET + TOKENIZER ====================
class Tokenizer:
    """Simple tokenizer from scratch. Hash words to indices"""
    def __init__(self, vocab_size=10000, max_len=77):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def encode(self, text):
        words = text.lower().split()[:self.max_len]
        idx = [hash(w) % self.vocab_size for w in words]
        idx += [0] * (self.max_len - len(idx)) # padding
        return torch.tensor(idx, dtype=torch.long)

class ImageTextDataset(Dataset):
    """
    Folder structure:
    /data/images/img_001.jpg
    /data/captions.txt -> "img_001.jpg a red car on street"
    """
    def __init__(self, img_folder, caption_file):
        self.img_folder = img_folder
        self.data = []

        # Load captions
        with open(caption_file, 'r') as f:
            for line in f:
                img_name, caption = line.strip().split(' ', 1)
                self.data.append((img_name, caption))

        # Image transforms
        self.transform = T.Compose([
            T.Resize(config["image_size"]),
            T.CenterCrop(config["image_size"]),
            T.ToTensor(),
            T.Normalize(0.5, 0.5) # Scale to [-1, 1] for diffusion
        ])
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_name, caption = self.data[i]
        img = Image.open(os.path.join(self.img_folder, img_name)).convert('RGB')
        img = self.transform(img)

        # Simple text encoder: embedding + mean pool
        text_idx = self.tokenizer.encode(caption)
        text_emb = nn.Embedding(10000, 512)(text_idx).mean(0) # [512]

        return img, text_emb

#==================== 5. TRAINING LOOP ====================
def train(model, dataloader, epochs=100, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(epochs):
        total_loss = 0
        for imgs, text_emb in dataloader:
            imgs, text_emb = imgs.to(device), text_emb.to(device)

            # 1. Sample random timestep for each image
            t = torch.randint(0, config["timesteps"], (imgs.shape[0],), device=device)

            # 2. Add noise: x_noisy = x + noise * alpha
            noise = torch.randn_like(imgs)
            alpha = t.float() / config["timesteps"]
            x_noisy = imgs + noise * alpha[:, None, None, None]

            # 3. Predict noise
            pred_noise = model(x_noisy, t.float(), text_emb)

            # 4. Loss = MSE between predicted and real noise
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), f"dit_image_epoch{epoch}.pth")

#==================== 6. INFERENCE - DDIM SAMPLING ====================
@torch.no_grad()
def generate(model, text_prompt, steps=50, device='cuda'):
    """
    Generate image from text. Start with random noise, denoise step by step
    """
    model.eval()
    tokenizer = Tokenizer()
    text_emb = nn.Embedding(10000, 512)(tokenizer.encode(text_prompt)).mean(0).unsqueeze(0).to(device)

    # Start with random noise
    x = torch.randn(1, 3, config["image_size"], config["image_size"]).to(device)

    # Denoise gradually
    step_size = config["timesteps"] // steps
    for t in reversed(range(0, config["timesteps"], step_size)):
        t_tensor = torch.tensor([t], device=device).float()
        pred_noise = model(x, t_tensor, text_emb)
        x = x - pred_noise * 0.1 # Simplified DDIM step

    # Denormalize from [-1,1] to [0,1]
    x = (x + 1) / 2
    return x.clamp(0, 1)


#==================== RUN ====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ImageDiT()

    # TRAINING
    # dataset = ImageTextDataset('data/images', 'data/captions.txt')
    # dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    # train(model, dataloader, epochs=100, device=device)

    # INFERENCE EXAMPLE
    # model.load_state_dict(torch.load('dit_image_epoch99.pth'))
    # img = generate(model, "a red sports car on highway", device=device)
    # T.ToPILImage()(img[0]).save('output.png')

