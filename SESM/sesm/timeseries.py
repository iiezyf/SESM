import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import torchsnooper


from sesm import (
    L2MHSA,
    MultiHeadSelectiveAttention,
    RNNAllOut,
    RNNHidden,
    GumbelSigmoid,
    ConvNormPool,
    Swish,
    UnfoldEmbed,
)


class Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_embed,
        n_heads,
        d_hidden,
        d_kernel,
        n_layers,
        d_out,
        max_len,
        dropout=0.2,
        **kwargs
    ):
        super().__init__()
        assert d_input == 1, "time-series requires d_input==1"
        self.d_kernel = d_kernel
        self.d_hidden = d_hidden
        self.n_heads = n_heads

        # embed
        # self.embed = ConvNormPool(1, d_hidden, d_kernel)
        # self.pool = nn.MaxPool1d(kernel_size=2)
        self.embed = nn.Sequential(
            nn.Conv1d(d_input, d_embed, d_kernel),
            nn.BatchNorm1d(num_features=d_embed),
            Swish(),
        )

        # self.embed = nn.Sequential(
        #     nn.LSTM(d_input, d_embed, n_layers, batch_first=True), RNNAllOut()
        # )

        ## sequential selector
        self.sequential_selector = MultiHeadSelectiveAttention(
            n_heads, d_embed, dropout
        )
        self.selected_encoder = nn.Sequential(
            ConvNormPool(d_embed, d_hidden, d_kernel),
            ConvNormPool(d_hidden, d_hidden, d_kernel),
            nn.AdaptiveMaxPool1d((1)),
        )
        # (Batch, Heads, Hidden)

        ## concept selector
        # self.concept_selector = nn.Sequential(
        #     encoder,
        #     nn.Dropout(dropout),
        #     nn.Linear(d_hidden // 2, n_heads),
        #     # nn.Tanh(),
        #     nn.Softmax(-1),
        # )
        self.concept_selector = nn.Sequential(
            ConvNormPool(d_embed, d_hidden, d_kernel),
            ConvNormPool(d_hidden, d_hidden, d_kernel),
            nn.AdaptiveMaxPool1d((1)),
        )
        self.proj = nn.Sequential(
            nn.BatchNorm1d(d_hidden),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_heads),
            # nn.Softmax(-1),
            nn.Softplus(),
        )
        # (Batch, Heads)

        ## concept aggregator and mlp output
        self.fc = nn.Sequential(
            nn.BatchNorm1d(d_hidden),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def classifier(self, sentence, sent_mask=None):
        x = self.embed(sentence.unsqueeze(1))
        # (Batch, Hidden, Seqlen//2)
        x = self.selected_encoder(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    # @torchsnooper.snoop()
    def forward(self, sentence, sent_mask=None, heads=False):
        """
        sentence: (Batch, Seqlen)
        """

        ## embed
        x = self.embed(sentence.unsqueeze(1)).transpose(-2, -1)
        # x = self.embed(sentence.unsqueeze(-1))
        # (Batch, Seqlen, Embed)
        if sent_mask is not None:
            # sent_mask = self.pool(sent_mask.unsqueeze(1).float()).bool().squeeze(1)
            # (Batch, Seqlen)
            sent_mask = sent_mask.unfold(-1, self.d_kernel, 1).any(-1)

        ## sequential selector
        sequential_selector = self.sequential_selector(x, sent_mask)
        (Batch, Heads, Seqlen) = sequential_selector.shape
        # (Batch, n_heads, Seqlen)
        x_sequential_selected = x.unsqueeze(1) * sequential_selector.unsqueeze(-1)
        x_sequential_selected = x_sequential_selected.reshape(
            Batch * Heads, Seqlen, -1
        ).transpose(-2, -1)
        # (Batch * Heads, Hidden, Seqlen)

        ## sequential encoder
        x_sequential_selected_encoded = self.selected_encoder(
            x_sequential_selected
        ).reshape(Batch, Heads, -1)
        # x_sequential_selected_encoded = F.layer_norm(
        #     x_sequential_selected_encoded, (self.d_hidden // 2,)
        # )
        # (Batch, Heads, Hidden // 2)

        # sequential

        ## concept selector
        concept_selector = self.concept_selector(x.transpose(-2, -1)).reshape(Batch, -1)
        concept_selector = self.proj(concept_selector)
        # (Batch, Heads)
        # concept_selector = torch.ones((Batch, Heads)).to(x.device)
        # (Batch, Heads)

        ## concept aggregator
        x_concept_selected = x_sequential_selected_encoded * concept_selector.unsqueeze(
            -1
        )
        # (Batch, Heads, Hidden)

        # x_concept_concatenated = x_concept_selected.view(Batch, -1)
        # (Batch, Head * Hidden)
        # x_concept_concatenated = x_concept_selected.mean(1)
        # (Batch, Hidden)

        ## output
        out = self.fc(x_sequential_selected_encoded.reshape(Batch * Heads, -1))
        # (Batch, Hidden)
        if not heads:
            out = (out.reshape(Batch, Heads, -1) * concept_selector.unsqueeze(-1)).sum(
                1
            )

        # print(sequential_selector.shape)
        # print(concept_selector.shape)

        ## 约束
        L_diversity = self._diversity_term(sequential_selector)
        L_fidelity = self._fidelity_term(
            x_sequential_selected_encoded, concept_selector
        )
        L_locality = self._locality_term(sequential_selector, sent_mask)
        L_simplicity = self._simplicity_term(concept_selector)

        return (
            out,
            [L_diversity, L_fidelity, L_locality, L_simplicity],
            sequential_selector,
            concept_selector,
        )

    def _diversity_term(self, x, d="euclidean", eps=1e-9):

        if d == "euclidean":
            # euclidean distance
            D = torch.cdist(x, x, 2)
            Rd = torch.relu(-D + 2)

            zero_diag = torch.ones_like(Rd, device=Rd.device) - torch.eye(
                x.shape[-2], device=Rd.device
            )
            return ((Rd * zero_diag)).sum() / 2.0

        elif d == "cosine":
            # cosine distance
            x_n = x.norm(dim=-1, keepdim=True)
            x_norm = x / torch.clamp(x_n, min=eps)
            D = 1 - torch.matmul(x_norm, x_norm.transpose(-1, -2))
            zero_diag = torch.ones_like(D, device=D.device) - torch.eye(
                x.shape[-2], device=D.device
            )
            return (D * zero_diag).sum() / 2.0

        else:
            raise NotImplementedError

    def _fidelity_term(self, x_sequential_selected_encoded, concept_selector):

        x = x_sequential_selected_encoded.transpose(0, 1)
        x_n = x.norm(dim=-1, keepdim=True)
        x_norm = x / torch.clamp(x_n, min=1e-9)
        D = 1 - torch.matmul(x_norm, x_norm.transpose(-1, -2))

        # D = torch.cdist(x, x, 2)  # (Head, Batch, Batch)

        mask = concept_selector.transpose(0, 1)  # (Head, Batch)
        mask = mask.unsqueeze(-2) * mask.unsqueeze(-1)  # (Head, Batch, Batch)

        return (D * mask).sum() / 2.0

        # (Head, Batch, Hidden)
        return torch.cdist(x, x, 2).sum() / 2.0

    # def _fidelity_term(self, sequential_selector, sent_mask):
    #     if sent_mask is not None:
    #         mask_pad = 1.0 - sent_mask.float()
    #     else:
    #         mask_pad = 0.0

    #     # selective attention fidility term
    #     norm_type = 1
    #     # (Batch, n_heads, Seqlen)
    #     Rd = torch.relu(1 - (sequential_selector.sum(1).squeeze(-1) + mask_pad))
    #     return torch.norm(Rd, p=norm_type, dim=1).sum()

    def _locality_term(self, sequential_selector, sent_mask):
        if sent_mask is not None:
            return (sequential_selector.sum(2) / sent_mask.unsqueeze(1).sum(2)).sum()
        else:
            return (sequential_selector.mean(2)).sum()

    def _simplicity_term(self, concept_selector):
        # (Batch, n_heads)
        return (concept_selector.sum(1) / concept_selector.shape[1]).sum()
