import math

from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
import torch

from cs336_basics.transformer.model import Linear, softmax, RotaryPositionalEmbedding


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    attn_score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        attn_score = attn_score.masked_fill(~mask, -torch.inf)
    attn_score = softmax(attn_score, dim=-1)
    output = einsum(attn_score, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return output

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device

        self.q_proj_weight = Linear(d_model, d_model, device)
        self.k_proj_weight = Linear(d_model, d_model, device)
        self.v_proj_weight = Linear(d_model, d_model, device)
        self.o_proj_weight = Linear(d_model, d_model, device)

        self.theta = theta
        self.max_seq_len = max_seq_len
        if self.theta and self.max_seq_len:
            self.rope = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len, self.device)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None
    ):
        x = x.to(self.device)
        Q = self.q_proj_weight(x)
        K = self.k_proj_weight(x)
        V = self.v_proj_weight(x)

        Q = rearrange(Q, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        K = rearrange(K, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        V = rearrange(V, "... s (h d_k) -> ... h s d_k", h=self.num_heads)

        s = x.shape[-2]
        causal_mask = torch.tril(torch.ones(s, s)).bool().to(self.device)
        if self.theta and self.max_seq_len:
            if token_positions is None:
                token_positions = torch.arange(s, dtype=torch.int).to(self.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        output = scaled_dot_product_attention(Q, K, V, causal_mask)
        output = rearrange(output, "... h s d_k -> ... s (h d_k)")
        output = self.o_proj_weight(output)

        return output