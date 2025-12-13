from einops import rearrange, reduce
from torch import Tensor
import torch


def cross_entropy(
    inputs: torch.Tensor, # (..., sequence_length, vocab_size)
    targets: torch.Tensor, # (..., sequence_length)
    device: str = None
):
    if device is not None:
        inputs = inputs.to(device)
        targets = targets.to(device)

    inputs_max = reduce(inputs, "... vocab_size -> ... 1", "max")
    stable_inputs = inputs - inputs_max

    log_sum_exp = reduce(stable_inputs.exp(), "... vocab_size -> ... 1", "sum").log()
    target_logits = stable_inputs[torch.arange(inputs.shape[0]), targets]

    ce_loss = log_sum_exp - target_logits

    return ce_loss.mean()