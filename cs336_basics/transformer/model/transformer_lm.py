from torch import Tensor, nn

from cs336_basics.transformer.model import Linear, Embedding, RMSNorm, CausalMultiHeadAttention, PositionWiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=None, theta=None, device=None):
        super().__init__()

        self.d_model = d_model
        self.device = device
        self.ln1 = RMSNorm(d_model, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.attn = CausalMultiHeadAttention(d_model, num_heads, max_seq_len, theta, device)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, device)
    
    def forward(self, x: Tensor):
        x = x.to(self.device)
        x = self.attn(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x

        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        device: str = None
    ):
        super().__init__()
        self.device = device

        self.token_embeddings = Embedding(vocab_size, d_model, device)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device)

    def forward(self, x: Tensor):
        x = x.to(self.device)
        x = self.token_embeddings(x)

        for attn_block in self.layers:
            x = attn_block(x)

        logits = self.lm_head(self.ln_final(x))

        return logits