from torch import nn
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):

        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x): # torch.Size([128, 20])
        tok_emb = self.tok_emb(x) # torch.Size([128, 20, 512])
        pos_emb = self.pos_emb(x) # torch.Size([20, 512])
        pos_emb = pos_emb.unsqueeze(0).expand(tok_emb.shape[0],-1,-1)
        return self.drop_out(tok_emb + pos_emb)   # torch.Size([128, 20, 512])
