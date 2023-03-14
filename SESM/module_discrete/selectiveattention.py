import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper
from sesm import GumbelSigmoid


class MultiHeadSelectiveAttention(nn.Module):
    def __init__(self, n_heads, d_embed, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadSelectiveAttention, self).__init__()
        # assert d_embed % n_heads == 0
        self.d_head = d_embed // n_heads
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_embed, self.d_head * self.n_heads)
        self.w_k = nn.Linear(d_embed, self.d_head * self.n_heads)
        self.w_v = nn.Linear(d_embed, n_heads)

        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.gumbel = GumbelSigmoid()

    # @torchsnooper.snoop()
    def forward(self, x, mask=None):
        # return x.unsqueeze(1).repeat(1, self.n_heads, 1, 1), torch.ones(x.shape[0], self.n_heads, x.shape[1]).to(x.device), torch.ones(x.shape[0], self.n_heads).to(x.device)

        # x (Batch, Seqlen, d_embed)

        nbatches = x.size(0)
        # 1) qkv分头线性映射
        # (Batch, n_heads, Seqlen, d_head)
        query = (
            self.w_q(x).view(nbatches, -1, self.n_heads, self.d_head).transpose(1, 2)
        )
        key = self.w_k(x).view(nbatches, -1, self.n_heads, self.d_head).transpose(1, 2)
        # (Batch, n_heads, Seqlen, 1)
        value = self.w_v(x).view(nbatches, -1, self.n_heads, 1).transpose(1, 2)

        # print(query.shape, key.shape, value.shape)

        # 2) 选择性(0-1) selective attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)
        # (Batch, n_heads, Seqlen, Seqlen)
        p_attn = self.gumbel(torch.matmul(scores, value)).squeeze(-1)
        # (Batch, n_heads, Seqlen)

        if mask is not None:
            p_attn = p_attn.masked_fill(~mask.unsqueeze(1), 0.0)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        return p_attn
