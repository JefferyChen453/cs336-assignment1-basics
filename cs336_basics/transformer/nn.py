from torch import nn, Tensor
from torch.nn import functional as F
import torch
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype))
        
        self.initialize()
    
    def initialize(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype))

        self.initialize()
    
    def initialize(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor):
        return self.weight[token_ids]