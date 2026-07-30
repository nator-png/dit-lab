import torch
import torch.nn as nn
import torch.nn.functional as F

text = "A quick brown Fox jumped over a silly lazy dog "
spl_txt = text.split()
flt_txt = sorted(set(spl_txt))

int2txt = {i:t for i,t in enumerate(flt_txt)}
txt2int = {t:i for i,t in enumerate(flt_txt)}

con =  {"embd":768,
        "con_length":8,
        "drop_rate":0.1,
        "num_layers":10,
        "num_heads":12,
        "head_dim":64,
        "qkv_bias":True,
        "batch_size":12,
        "vocab_size":len(flt_txt)
        }

data = torch.tensor([txt2int[t] for t in spl_txt])

def get_batch():
    ix = torch.randint(0,len(data)-con["con_length"]-1,(con["batch_size"],))
    x = torch.stack([data[i:i+con["con_length"]] for i in ix])
    y = torch.stack([data[i+1:i+con["con_length"]+1] for i in ix])

    return x,y

class Attention(nn.Module):
    def __init__(self,d_in = con["embd"],d_out = con["embd"],num_heads = con["num_heads"],head_dim = con["head_dim"],cont_len = con["con_length"],qkv_bias = con["qkv_bias"],drop_rate = con["drop_rate"]):
        super().__init__()

        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.drop_out = nn.Dropout(drop_rate)
        self.out_layer = nn.Linear(d_out,d_out,bias=qkv_bias)
        self.register_buffer("mask",torch.triu(torch.ones(con["con_length"],con["con_length"]),diagonal=1))
        self.heads = num_heads
        self.head_dim = head_dim


    def forward(self,x):
        batch,num_tok,embd = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(batch,num_tok,self.heads,self.head_dim)
        keys = keys.view(batch,num_tok,self.heads,self.head_dim)
        values = values.view(batch,num_tok,self.heads,self.head_dim)

        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        masked = self.mask.bool()[:num_tok,:num_tok]
        attn_scores.masked_fill_(masked,-torch.inf)
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1) 
        attn_weights = self.drop_out(attn_weights)

        cont_vec = (attn_weights @ values).transpose(1,2)
        cont_vec = cont_vec.contiguous().view(batch,num_tok,embd)
        cont_vec = self.out_layer(cont_vec)

        return cont_vec
    
class LayerNorm(nn.Module):
    def __init__(self,embd = con["embd"]):
        super().__init__()
        
        self.scale = nn.Parameter(torch.ones(embd))
        self.shift = nn.Parameter(torch.zeros(embd))
        self.eps = 1e-5

    def forward(self,x):
        
        mean = x.mean(keepdim = True,dim = -1)
        var = x.var(keepdim = True,dim = -1)
        out = (x-mean)/torch.sqrt(var+self.eps)

        return self.scale * out + self.shift
    


class FeedForward(nn.Module):
    def __init__(self,embd = con["embd"]):
        super().__init__()

        self.layer = nn.Sequential(nn.Linear(embd,embd*4),nn.GELU(),nn.Linear(embd*4,embd),)

    def forward(self,x):

        x = self.layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()
        self.drop_shortcut = nn.Dropout(con["drop_rate"])
        self.att = Attention()
        self.ff = FeedForward()
        
    def forward(self,x):
        
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = shortcut + x

        return x


class Model(nn.Module):
    def __init__(self,embd = con["embd"],vocab_size = con["vocab_size"],drop_rate = con["drop_rate"],con_length = con["con_length"]):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size,embd)
        self.pos_emb = nn.Embedding(con_length,embd)
        self.final_norm = LayerNorm()
        self.out_head = nn.Linear(embd,vocab_size,bias=True)
        self.trf = nn.Sequential(*[TransformerBlock() for _  in range(con["num_layers"])])
        self.drop_emb = nn.Dropout(drop_rate)

    def forward(self,x):
        batch,num_tok = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(num_tok))
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
    


NLP = Model()
optimizer = torch.optim.AdamW(NLP.parameters(),lr=5e-4)

steps = 50
for step in range(steps):
    x,y = get_batch()
    logits = NLP(x)
    loss = F.cross_entropy(logits.view(-1,con["vocab_size"]),y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 2 == 0:
        print(f"Step:{step}   Loss:{loss.item():.4f}")
    

def Generate(prompt,max_tok):
    words = torch.tensor([[txt2int[t] for t in prompt.split()]])
    
    for _ in range(max_tok):
            
            with torch.no_grad():    
                ids = words[:,-con["con_length"]:]
            logits = NLP(ids)[:,-1,:]
            prob = torch.softmax(logits,-1)
            next_id = torch.multinomial(prob,1)

            words = torch.cat((words,next_id),1)


    return [int2txt[int(i)] for i in words[0]]

print(Generate("silly",8))





        
