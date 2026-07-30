import torch
import random
import torchaudio
"""
def custom_collate_func(batch):

    max_len = max(b. for b in batch)

    return max_len


x = torch.tensor([1,2,3,4])
y = torch.tensor([5,6,7,8,9])
z = torch.tensor([10,11,12,13,14])

batch = (x,y,z)


print(custom_collate_func(batch))"""

aud,sr = torchaudio.load("C:/Users/block/Music/Fireboy_DML_&_D_Smoke_-_Champion_(Audio)(128k).mp3")
print(aud.shape)
print(sr)