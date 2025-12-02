from einops import reduce
from torch import Tensor, nn
import torch


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