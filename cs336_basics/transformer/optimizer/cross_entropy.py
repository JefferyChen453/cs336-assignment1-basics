from einops import rearrange, reduce
from torch import Tensor
import torch


def cross_entropy(
    inputs: Tensor, # logits
    targets: Tensor,
    device: str = None
):
    inputs = inputs.to(device)
    targets = targets.to(device)
    max_logits = reduce(inputs, "... vocab_size -> ... 1", "max")
    stable_inputs = inputs - max_logits
    log_exp_sum = reduce(stable_inputs.exp(), "... vocab_size -> ... 1", "sum").log()
    correct_logits = torch.gather(stable_inputs, dim=-1, index=rearrange(targets.long(), "batch_size ... -> batch_size 1 ..."))
    loss = - (correct_logits - log_exp_sum)

    return loss.mean()
