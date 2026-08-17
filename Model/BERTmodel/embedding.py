import torch.nn as nn
import torch
from math import cbrt,pi,ceil

import numpy as np


def get_M(angles):

    c1 = torch.cos(angles[...,0])
    s1 = torch.sin(angles[...,0])
    # print('c1',c1.shape, 's1', s1.shape)
    M = torch.stack([torch.stack([c1, -s1]),
                       torch.stack([s1, c1])])
    M = M.permute(2,3,0,1).float()
    return M

def rope_rotation(x, angles):
    '''
    x : torch.tensor of size *,4
    angles : (N ,t1) # *,1 angle in radians
    '''
    M = get_M(angles).to(x.device)
    x = x.unsqueeze(-1)
    # print('x',x.shape, 'M', M.shape)
    x = torch.matmul(M, x) # Encodes t1 and t3
    x = x.squeeze(-1)
    return x


class StandardRoPE(nn.Module):
    def __init__(self, embed_dim, base = 10000):
        super().__init__()

        req_dims = 2
        self.req_dims = req_dims
        self.d_model = embed_dim
        assert self.d_model % req_dims == 0, 'RoPE requires even dimentions'
        angles = ((np.random.randn((self.d_model//req_dims))*(2*pi))/base) # (d_model//2,)
        # print('angles',angles.shape)
        angles = torch.from_numpy(angles.reshape(1,-1)) # (1, d_model//2)
        indices = torch.from_numpy(np.arange(base)).unsqueeze(1) # (N, 1)
        # print('angles, indices', angles.shape, indices.shape)
        self.register_buffer('angles',(angles*(indices)).unsqueeze(-1)) # (N, d_model//2, 1)
        # print('angles_indices', self.angles.shape)
        # self.register_buffer('angles', angles) # (1, d_model//2)
    def forward(self, x):
        '''
        rotary embedding
        '''
        x_i = x.reshape(*x.shape[:-1],-1,self.req_dims) # ..., d_model//2, 2
        # print('x_i',x_i.shape, 'angles', self.angles.shape)
        x_i = rope_rotation(x_i, self.angles[:x_i.shape[1],...]) # ..., d_model//2, 2
        x = x_i.flatten(-2,-1)
        return x
    
class AbsolutePosition(nn.Module):
    def __init__(self, max_length, embedding_dim):
        super().__init__()

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim
        )

    def forward(self, x):
        # x: [batch_size, sequence_length, embedding_dim]
        batch_size, sequence_length, _ = x.shape

        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        return x + self.position_embedding(positions)

class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size= 100000, embed_dim=512):
        '''
        pre norm embedder
        '''
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        # x has the token number (B, N, 1)
        return self.embed(x.squeeze(-1))

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_dim, dropout=0.1, base = 10000):
        super().__init__()
        '''
        full_image_size = [H,W,D]
        patch_size = [pH,pW,pD]
        inp_size = np.prod(patch_size)
        embed_size = embedding size
        block_size = [H/pH, W/pW, D/pD] Number of patches along each dimension
        '''
        # print('pss',patch_size)
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
        self.position = StandardRoPE(embed_dim=embed_dim, base=base)
        self.absolute = AbsolutePosition(max_length=1000, embedding_dim=embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        t = self.absolute(self.token(x))
        t = self.position(t)
        return self.dropout(t)


# tests
# b = BERTEmbedding(100000, 720, base = 10000)
# x = torch.randint(0,100000,(8, 10000, 1))
# print(torch.max(x), torch.min(x))
# o = b(x)
# print(o.shape)