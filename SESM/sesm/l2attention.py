import torch
from torch import norm
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torchsnooper import snoop

from sesm import GumbelSigmoid


class L2MHSA(nn.Module):
    def __init__(self, n_heads: int, d_hidden: int, dropout=None) -> None:
        """L2 Self-Attention with Selective Mechanism

        Args:
            d_hidden (int): hidden size
            n_heads (int): num of heads
            dropout (float, optional): dropout rate. Defaults to 0.1.
        """
        super().__init__()
        assert (
            d_hidden % n_heads == 0
        ), f"Hidden must be divisible by Heads, {d_hidden, n_heads}"
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        self.d_hidden = d_hidden
        self.dropout = 0.1

        self.Wq = nn.Parameter(torch.zeros(d_hidden, d_hidden))
        self.Wv = nn.Parameter(torch.zeros(d_hidden, 1))

        nn.init.kaiming_normal_(self.Wq)
        nn.init.kaiming_normal_(self.Wv)

        self.dropout = nn.Dropout(dropout) if dropout else None

        self.gumbel = GumbelSigmoid()

    # @snoop()
    def forward(self, X, mask=None):
        """forward

        Args:
            X (torch.FloatTensor): (Batch, Seqlen, Hidden)
            mask (torch.BoolTensor, optional): (Batch, Seqlen), False for masked out. Defaults to None.
        """

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)
            # (Batch, 1, Seqlen, 1)

        Batch, Seqlen = X.shape[:2]

        Q = (
            torch.matmul(X, self.Wq)
            .reshape(Batch, Seqlen, self.n_heads, self.d_head)
            .transpose(1, 2)
            .reshape(Batch * self.n_heads, Seqlen, self.d_head)
        )
        # print(Q.shape)

        # the attention
        P = F.softmax(
            -(
                norm(Q, p=2, dim=-1, keepdim=True)
                - 2 * torch.matmul(Q, Q.transpose(-2, -1))
                + norm(Q, p=2, dim=-1, keepdim=True).transpose(-2, -1)
            )
            / sqrt(self.d_head),
            dim=-1,
        ).reshape(Batch, self.n_heads, Seqlen, Seqlen)

        Wh = self.Wq.reshape(self.d_hidden, self.n_heads, self.d_head).transpose(0, 1)
        Ah = torch.matmul(Wh, Wh.transpose(-2, -1))
        # (n_heads, Hidden, Hidden)

        attn_scores = torch.matmul(torch.matmul(P, X.unsqueeze(1)), Ah)
        # (Batch, n_heads, Seqlen, Hidden)

        applied = torch.matmul(attn_scores, self.Wv)
        # (Batch, n_heads, Seqlen, 1)

        # applied = applied.transpose(1, 2).reshape(Batch, Seqlen, -1)
        # (Batch, Seqlen, Hidden)

        selection = self.gumbel(applied)

        if self.dropout:
            selection = self.dropout(selection)

        if mask is not None:
            selection.masked_fill(mask, 0.0)

        return selection
