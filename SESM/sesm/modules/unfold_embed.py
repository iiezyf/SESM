import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class UnfoldEmbed(nn.Module):
    def __init__(self, d_kernel, hidden_size, max_len, dropout=0.1) -> None:
        super().__init__()
        assert d_kernel % 2 == 1, "kernel size must be odd"
        self.d_kernel = d_kernel
        self.embed = nn.Linear(d_kernel, hidden_size)
        # self.pos_embed = PositionalEncoding(hidden_size, dropout, max_len)

    def forward(self, x):
        x = F.pad(x, (self.d_kernel // 2, self.d_kernel // 2))
        x = x.unfold(-1, self.d_kernel, 1)
        x = self.embed(x)
        # x = self.pos_embed(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
