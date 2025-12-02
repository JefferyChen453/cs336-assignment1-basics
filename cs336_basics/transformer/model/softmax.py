from torch import Tensor
import torch


def softmax(x: Tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max

    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    return exp_x / sum_exp