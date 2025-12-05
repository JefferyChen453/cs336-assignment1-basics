import numpy as np
import numpy.typing as npt
import torch


def get_batch_data(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
):
    total_len = dataset.shape[0]
    max_start_idx = total_len - context_length
    start_idxes = np.random.randint(0, max_start_idx, size=batch_size)
    
    samples = torch.stack(
        [torch.tensor(dataset[start_idx : start_idx + context_length], dtype=torch.int) for start_idx in start_idxes],
        dim=0
    ).to(device)
    targets = torch.stack(
        [torch.tensor(dataset[start_idx + 1: start_idx + context_length + 1], dtype=torch.int) for start_idx in start_idxes],
        dim=0
    ).to(device)

    return samples, targets


def read_nparray(file_path):
    return np.memmap(file_path, dtype=np.uint16, mode="r")