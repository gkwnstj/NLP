import math

from torch import nn


class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size() # torch.Size([128, 8, 20, 64])

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose  # torch.Size([128, 8, 64, 20])
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product # q : (128, 8, 20, 64)  # torch.Size([128, 8, 20, 20])

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)   #torch.Size([128, 8, 20, 20])

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score) # torch.Size([128, 8, 20, 20])

        # 4. multiply with Value
        v = score @ v    # (128, 8, 20, 20) * v: (128, 8, 20, 64)

        return v, score   # (128, 8, 20, 64), (128, 8, 20, 20)
