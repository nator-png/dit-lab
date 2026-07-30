import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as T
import hashlib
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt




config = {
    "embd":256,
    "img_size":(256,256),
    "layers":10,
    "num_heads":16,
    "vocab_size":100000,
    "batch_size":1,
    "wrd_emb":64,
    "patch_size":16,
    "channels":3,
    "max_tok":100,
    "epochs":10,
    "steps":1000

}

class Embedding(nn.Module):
    def __init__(self,embd = config["embd"],patch_size = config["patch_size"],channels = config["channels"]):
        super().__init__()
        
        self.num_patches = (config["img_size"][0]//patch_size) * (config["img_size"][1]//patch_size)
        self.projection = nn.Conv2d(channels,embd,patch_size,patch_size)
        self.pos_emb = nn.Parameter(torch.rand(1,self.num_patches,embd))

    def forward(self,x):

        x = self.projection(x)
        b = x.shape[0]
        embd = x.shape[1]
        x = x.permute(0,2,3,1).reshape(b,self.num_patches,embd)
        x = x + self.pos_emb

        return x
    


class TimeStepEmb(nn.Module):
    def __init__(self,embd = config["embd"]):
        super().__init__()

        self.half_dim = embd//2

        self.mlp = nn.Sequential(nn.Linear(embd,embd*4),
                                 nn.SiLU(),
                                 nn.Linear(embd*4,embd))
        
    def forward(self,t):

        freqs = torch.exp(torch.arange(self.half_dim,dtype=torch.float32) * -np.log(10000)/(self.half_dim-1))

        emb = t[:,None].float() * freqs[None,:]
        emb = torch.cat([emb.sin(),emb.cos()],dim=-1)

        return(self.mlp(emb))
    


class DiTBlock(nn.Module):
    def __init__(self,embd = config["embd"],num_heads = config["num_heads"]):
        super().__init__()

        self.txt_crossatt = nn.MultiheadAttention(embed_dim=embd,num_heads=num_heads,batch_first=True)
        self.img_att = nn.MultiheadAttention(embed_dim=embd,num_heads=num_heads,batch_first=True)
        self.ff = nn.Sequential(nn.Linear(embd,embd*4),
                                nn.GELU(),
                                nn.Linear(embd*4,embd))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),
                                              nn.Linear(embd,embd*6))
        
        self.norm1 = nn.LayerNorm(embd)
        self.norm2 = nn.LayerNorm(embd)
        self.norm3 = nn.LayerNorm(embd)
        
    def forward(self,img_emb,text_emb,t):

        scale1,shift1,gate1,scale2,shift2,gate2 = self.adaLN_modulation(t).chunk(6,dim = -1)

        img_emb = self.norm1(img_emb) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        img_emb = img_emb + gate1.unsqueeze(1) * self.img_att(img_emb,img_emb,img_emb)[0]

        img_emb = self.norm2(img_emb) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        img_emb = img_emb + gate2.unsqueeze(1) * self.txt_crossatt(img_emb,text_emb,text_emb)[0]

        img_emb = self.norm3(img_emb)
        x = img_emb + self.ff(img_emb)

        return x
    




class DiTModel(nn.Module):
    def __init__(self,embd = config["embd"],patch_size = config["patch_size"],channels = config["channels"],layers = config["layers"],wrd_emb = config["wrd_emb"]):
        super().__init__()

        patch_dim = (patch_size**2)*channels
        self.projections = Embedding()
        self.t_emb = TimeStepEmb()
        self.dit_block = nn.ModuleList([DiTBlock() for block in range(layers)])
        self.final_norm = nn.Linear(embd,patch_dim)
        self.wrd_emb = nn.Linear(wrd_emb,embd)
        self.grid_h = (config["img_size"][0])//patch_size
        self.grid_w = (config["img_size"][1])//patch_size
        self.channels = config["channels"]
        self.patch_size = patch_size

    def forward(self,img,text,t):

        img_emb = self.projections(img)
        text = self.wrd_emb(text).unsqueeze(1)
        t_emb = self.t_emb(t)

        for block in self.dit_block:
            img_emb = block(img_emb,text,t_emb)
        
        x = self.final_norm(img_emb)
        b = x.shape[0]

        x = x.reshape(b,self.grid_h,self.grid_w,self.channels,self.patch_size,self.patch_size).permute(0,3,1,4,2,5).reshape(b,self.channels,config["img_size"][0],config["img_size"][1])
        return x
    






class Tokenizer:
    def __init__(self,vocab_size = config["vocab_size"],max_tok = config["max_tok"]):
        
        self.vocab_size = vocab_size
        self.max_tok = max_tok





    def encode(self,text):

        ids = ([int(hashlib.md5(word.encode()).hexdigest(),16)%config["vocab_size"] for word in text.split()[:self.max_tok]])
        ids +=  (int(self.max_tok - len(ids))) * [0]
        
        return torch.tensor(ids,dtype=torch.long)
    


class ImageText(Dataset):
    def __init__(self,img_path,caption_file_path):
        super().__init__()

        self.data = []
        self.img_folder = img_path
        with open(caption_file_path,"r") as f:

            for line in f:
                img_name,caption_name = line.strip().split(" ",1)
                self.data.append((img_name,caption_name))



        self.Tokenizer = Tokenizer()

        self.transform = T.Compose([
            T.Resize(config["img_size"]),
            T.CenterCrop(config["img_size"]),
            T.ToTensor(),
            T.Normalize(0.5,0.5)
        ])

    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, index):

        img_name,words = self.data[index]
        ids = self.Tokenizer.encode(words)
        text_emb = nn.Embedding(config["vocab_size"],config["wrd_emb"])(ids).mean(0)
        
        img_folder = os.path.join(self.img_folder,img_name)
        img_tensor = Image.open(img_folder).convert('RGB')


        img_tensor = self.transform(img_tensor)

        return text_emb,img_tensor
    


def train_func(model, dataloader, epochs = config["epochs"], lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

    for epoch in range(epochs):

        epoch_loss = 0

        for text,img_tensor in dataloader:

            optimizer.zero_grad()

            t = torch.randint(1,config["steps"],(config["batch_size"],))
            alpha = t/config["steps"]
            noisy_img = torch.randn_like(img_tensor)
            noisy_img = img_tensor + noisy_img * alpha[:,None,None,None]

            predictions = model(noisy_img,text,t)
            loss = F.mse_loss(predictions,img_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss
        
        print(f"Loss :=> {epoch_loss}")
            

    


@torch.no_grad()

def generate(prompt,model,steps=50):
    
    model.eval()
    img = torch.randn(config["batch_size"],config["channels"],config["img_size"][0],config["img_size"][1])
    text = Tokenizer().encode(prompt)
    text_emb = nn.Embedding(config["vocab_size"],config["wrd_emb"])(text).mean(0).unsqueeze(0)
    
    step_size = config["steps"]//steps


    for t in reversed(range(1,config["steps"],step_size)):

        t = torch.tensor([t])

        predictions = model(img,text_emb,t)
        img = img  - predictions * 0.1
    
    img = (img + 1)/2
    img = img.clamp(0,1)

    img = img.reshape(config["img_size"][0],config["img_size"][1],config["channels"]).numpy()

    plt.imshow(img)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    dataset = ImageText("data/images", "data/captions.txt")
    model = DiTModel()
    loader = DataLoader(dataset=dataset,batch_size=config["batch_size"])
    train_func(model=model, dataloader=loader)

