from torch import nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # (128,20,512), (128,20,512), (128,20,512)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)  # (128,8,20,64), (128,8,20,64), (128,8,20,64) 

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)   # (128, 8, 20, 64), (128, 8, 20, 20)

        # 4. concat and pass to linear layer
        out = self.concat(out)   # 128,20,512
        out = self.w_concat(out)   # 128,20,512

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor): # torch.Size([128, 20, 512])
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size() # torch.Size([128, 20, 512])

        d_tensor = d_model // self.n_head # 64//8  # // 연산자는 나눗셈 연산 후 소수점 이하를 버리고 정수 부분만을 반환
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2) # torch.Size([128, 8, 20, 64])
        # it is similar with group convolution (split by number of heads)

        return tensor   # torch.Size([128, 8, 20, 64])

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()  # 128,8,20,64
        d_model = head * d_tensor # 512

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor   # torch.Size([128, 20, 512])
