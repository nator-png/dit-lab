import torch

x = torch.rand(6,8)
print(x.shape)
y = x.unfold(0,2,2)
print(y.shape)