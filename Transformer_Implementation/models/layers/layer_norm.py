import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):   # torch.Size([128, 20, 512])
        mean = x.mean(-1, keepdim=True)   # torch.Size([128, 20, 1])
        var = x.var(-1, unbiased=False, keepdim=True) # torch.Size([128, 20, 1])
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out   # torch.Size([128, 20, 512])
