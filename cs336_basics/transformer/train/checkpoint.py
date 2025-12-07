import os
import typing

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    save_model_only: bool = True
):
    os.makedirs(out, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "iteration": iteration,
    }
    if not save_model_only:
        checkpoint["optimizer"] = optimizer.state_dict()
    out = os.path.join(out, f"iter_{iteration:05d}.bin")
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
