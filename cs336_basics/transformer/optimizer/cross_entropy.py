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
    correct_logits = torch.gather(stable_inputs, dim=-1, index=rearrange(targets, "batch_size ... -> batch_size 1 ..."))
    loss = - reduce((correct_logits - log_exp_sum), "... vocab_size -> ... 1", "sum")

    return loss.mean()

def compute_perplexity(
    ce_losses: Tensor
):
    return reduce(ce_losses, "... seq_len -> ... 1", "mean").exp().mean()