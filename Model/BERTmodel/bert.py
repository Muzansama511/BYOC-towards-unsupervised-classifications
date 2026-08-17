# from sympy import use
from math import cbrt
import torch.nn as nn
import torch
import numpy as np
try:
    from .transformer import TransformerBlock
    from .embedding import BERTEmbedding
except:
    from transformer import TransformerBlock
    from embedding import BERTEmbedding
    

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, base = 2000, use_deepnet_init: bool = True, encoder: bool = True):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.use_deepnet_init = use_deepnet_init
        self.feed_forward_hidden = hidden * 4
        self.encoder = encoder
        if encoder:
            self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_dim=hidden, dropout=dropout, base=base)  # Embedding layer to convert input patches to hidden size

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, use_deepnet_init=self.use_deepnet_init, num_layers = n_layers) for _ in range(n_layers)])
        if not encoder:
            self.predicter = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, vocab_size)
            )
    def forward(self, x, mask = None):
        # x.shape = B, N
        # mask = B, N
        if self.encoder:
            x = self.embedding(x)
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i](x,mask=mask)
        if not self.encoder:
            x = self.predicter(x)
        return x

# # tests
# # # print('GPU recognised:', torch.cuda.is_available())
# # # exit()
# # device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # base = 2000
# # b = BERT(vocab_size=100000, hidden=128, n_layers=4, attn_heads=2, dropout=0.1, base=base).to(device)
# # optim = torch.optim.Adam(b.parameters(), 0.01)

# # x = torch.randint(0,100000,(16, base, 1)).to(device)
# # o = b(x)
# # optim.zero_grad()
# # loss = o.sum()
# # loss.backward()
# # optim.step()

# # for masks

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# base = 2000
# b = BERT(vocab_size=100000, hidden=128, n_layers=4, attn_heads=2, dropout=0.1, base=base).to(device)
# optim = torch.optim.Adam(b.parameters(), 0.01)

# x = torch.randint(0,100000,(16, base, 1)).to(device)
# mask = torch.randint(0,2,(16, 1, base)).to(device)
# mask[...,0] = 0
# o = b(x, mask) # passed visual check, need to print out mask and p_attn
# optim.zero_grad()
# loss = o.sum()
# loss.backward()
# optim.step()


# print(o.shape)

