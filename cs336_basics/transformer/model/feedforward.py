from torch import nn
import torch

from cs336_basics.transformer.model import Linear


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