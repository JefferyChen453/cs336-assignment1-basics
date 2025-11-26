import math

from einops import einsum, reduce, rearrange, repeat
from torch import Tensor, nn
import torch

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

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(reduce(x_fp32 ** 2, "b s d -> b s 1", "mean") + self.eps)
        ret = x_fp32 / rms * self.weight

        return ret.to(x.dtype)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.W_1 = Linear(d_model, d_ff, device, dtype)
        self.W_2 = Linear(d_ff, d_model, device, dtype)
        self.W_3 = Linear(d_model, d_ff, device, dtype)

    @staticmethod
    def SiLU(x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        """SwiGLU"""
        ret = self.SiLU(self.W_1(x)) * self.W_3(x)
        ret = self.W_2(ret)

        return ret

class RotaryPositionalEmbedding(nn.Module):
    """Source: https://spaces.ac.cn/archives/8265/comment-page-1"""
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.create_positional_embedding()
    
    def create_positional_embedding(self):
        freq_list = [pow(self.theta, - (2 * (i // 2)) / self.d_k) for i in range(self.d_k)]
        freq_list = torch.tensor(freq_list).to(self.device) # (d_k, )
        pos_list = torch.arange(self.max_seq_len, device=self.device).to(self.device) # (max_seq_len,)
        cos_rot_matrix = einsum(freq_list, pos_list, "d_k, max_seq_len -> max_seq_len d_k").cos()
        sin_rot_matrix = einsum(freq_list, pos_list, "d_k, max_seq_len -> max_seq_len d_k").sin()
        R = torch.stack([cos_rot_matrix, sin_rot_matrix], dim=1) # (max_seq_len, 2, d_k)

        self.register_buffer("R", R, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        R = repeat(self.R, "... -> b ...", b=x.shape[0]) # (b, max_seq_len, 2, d_k)
        token_positions = repeat(token_positions, "... -> b ...", b=x.shape[0])
        R = self.R[token_positions] 
        x_half1 = x[..., 0::2] # (q0 q2 q4 ...)
        x_half2 = -x[..., 1::2] # (-q1 -q3 -q5 ...)
        x_ = torch.stack([x_half2, x_half1], dim=-1).flatten(start_dim=-2) # (-q1 q0 -q3 q2 ...)
        rot_1, rot_2 = R[:, :, 0, :], R[:, :, 1, :]
        ret = x * rot_1 + x_ * rot_2

        return ret

def softmax(x: Tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max

    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    return exp_x / sum_exp



if __name__ == "main":
    device = "cuda"
    rope = RotaryPositionalEmbedding(10000, 64, 100, device=device)
    x = torch.arange(4*50*64).view((4, 50, 64)).to(device)
    token_positions = torch.arange(50, device=x.device, dtype=torch.int32)

    print(rope(x, token_positions))
