import torch.nn as nn

try:
    from .attention import MultiHeadedAttention
except:
    from attention import MultiHeadedAttention
    
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch





class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = LayerNorm(d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.norm(self.w_1(x)))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, use_deepnet_init: bool = True, num_layers: int = 12):
        super(SublayerConnection, self).__init__()
        self.use_deepnet_init = use_deepnet_init
        self.alpha = 1.0 if not use_deepnet_init else 2*num_layers**(0.25)
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if not self.use_deepnet_init:
            return x * self.alpha + self.dropout(sublayer(self.norm(x))) # pre_norm
        else:
            return self.norm(x * self.alpha + self.dropout(sublayer((x)))) # post_norm
            
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, use_deepnet_init: bool = True, num_layers: int = 12):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, use_deepnet_init=use_deepnet_init, num_layers=num_layers)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout, use_deepnet_init=use_deepnet_init, num_layers=num_layers)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout, use_deepnet_init=use_deepnet_init, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # print('x',x.shape)
        x = self.input_sublayer(x, lambda _x: checkpoint(self.attention.forward, _x, _x, _x, mask, use_reentrant=True))
        # x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
