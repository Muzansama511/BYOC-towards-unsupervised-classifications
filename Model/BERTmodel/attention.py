import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.checkpoint import checkpoint
import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        # print(scores.shape, mask.shape)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        # print(mask.unsqueeze(1) == 0)
        # print(mask)
        # print(p_attn)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, use_deepnet_init=False, num_layers=12):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.beta = 1.0 if not use_deepnet_init else (8 * num_layers** (-0.25)) # Deepnet scalar
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        for i in range(3):
            nn.init.xavier_normal_(self.linear_layers[i].weight, gain=self.beta)
        self.output_linear = nn.Linear(d_model, d_model)
        nn.init.xavier_normal_(self.output_linear.weight, gain=self.beta)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout) 
        # x,attn = checkpoint(self.attention, query, key, value, mask, self.dropout, use_reentrant=True)
        # print('attn',attn.shape)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
