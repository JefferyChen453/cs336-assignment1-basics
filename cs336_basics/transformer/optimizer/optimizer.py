import math
from typing import Optional, Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, **kwargs):
        defaults = {
            "lr": lr,
            "betas": tuple(betas),
            "eps": eps,
            "weight_decay": weight_decay
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data

                if "m" not in state:
                    state["m"] = torch.zeros_like(grad)
                if "v" not in state:
                    state["v"] = torch.zeros_like(grad)
                m_t = state["m"]
                v_t = state["v"]

                # in-place computation
                m_t.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                v_t.mul_(beta_2).add_(grad.pow(2), alpha=1 - beta_2)
                lr_t = lr * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                
                p.data.addcdiv_(m_t, (v_t.sqrt() + eps), value=-lr_t)
                p.data.add_(p.data, alpha=- lr * weight_decay)

                state["t"] = t + 1
                
        
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        lr = min_learning_rate + \
            0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate
    
    return lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], clip: float, eps: float = 1e-6):
    general_norm = 0
    for p in parameters:
        if p.grad is not None:
            general_norm += p.grad.data.norm(2) ** 2
    general_norm = general_norm ** 0.5
    if general_norm > clip:
        norm_coef = clip/(general_norm+eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(norm_coef)

    return general_norm