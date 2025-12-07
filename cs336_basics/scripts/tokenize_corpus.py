import multiprocessing
from multiprocessing import Pool
import os
from typing import Tuple

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.utils.file import stream_chunks_with_split

# global tokenizer instance for each worker
GLOBAL_TOKENIZER = None


def init_tokenizer(vocab_path, merges_path, special_tokens):
    """Executed once per worker process."""
    global GLOBAL_TOKENIZER
    GLOBAL_TOKENIZER = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=special_tokens,
    )


def process_chunk(args):
    global GLOBAL_TOKENIZER
    chunk_data, output_folder, idx = args

    text = chunk_data.decode("utf-8", errors="ignore")
    ids = GLOBAL_TOKENIZER.encode(text)

    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(os.path.join(output_folder, f"{idx:05d}.bin"))


def tokenize_parallel(
    vocab_path,
    merges_path,
    special_tokens,
    input_path: str,
    output_folder: str,
    num_workers: int = 8,
):
    os.makedirs(output_folder, exist_ok=True)
    block_size = 32 * 1024 * 1024  # 256MB chunk

    # create worker processes with tokenizer initialized inside
    pool = Pool(
        num_workers,
        initializer=init_tokenizer,
        initargs=(vocab_path, merges_path, special_tokens),
    )

    futures = []

    for idx, chunk in enumerate(
        tqdm(
            stream_chunks_with_split(
                input_path,
                block_size=block_size,
                reserve_split_token=True
            ),
            desc=f"Tokenizing {input_path} ..."
        )
    ):
        futures.append(
            pool.apply_async(
                process_chunk,
                args=((chunk, output_folder, idx),)
            )
        )

    for f in futures:
        f.get()

    pool.close()
    pool.join()


def main():
    num_processes = multiprocessing.cpu_count()

    # ---------------- TinyStories ----------------
    tinystories_vocab = "/workspace/cs336-assignment1-basics/data/tokenizer_TinyStories_10k/vocab.json"
    tinystories_merges = "/workspace/cs336-assignment1-basics/data/tokenizer_TinyStories_10k/merges.txt"
    special = ["<|endoftext|>"]

    tokenize_parallel(
        tinystories_vocab,
        tinystories_merges,
        special,
        "/workspace/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        "/workspace/cs336-assignment1-basics/data/data_bin/tinystories_train",
        num_processes,
    )

    tokenize_parallel(
        tinystories_vocab,
        tinystories_merges,
        special,
        "/workspace/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
        "/workspace/cs336-assignment1-basics/data/data_bin/tinystories_valid",
        num_processes,
    )

    # ---------------- OWT ----------------
    owt_vocab = "/workspace/cs336-assignment1-basics/data/tokenizer_owt_32k_optim/vocab.json"
    owt_merges = "/workspace/cs336-assignment1-basics/data/tokenizer_owt_32k_optim/merges.txt"

    tokenize_parallel(
        owt_vocab,
        owt_merges,
        special,
        "/workspace/cs336-assignment1-basics/data/owt_train.txt",
        "/workspace/cs336-assignment1-basics/data/data_bin/owt_train",
        num_processes,
    )

    tokenize_parallel(
        owt_vocab,
        owt_merges,
        special,
        "/workspace/cs336-assignment1-basics/data/owt_valid.txt",
        "/workspace/cs336-assignment1-basics/data/data_bin/owt_valid",
        num_processes,
    )


if __name__ == "__main__":
    # main()
    file_path = '/workspace/cs336-assignment1-basics/data/data_bin/owt_train/00350.bin'
    arr = np.fromfile(file_path, dtype=np.uint16)
    print(arr)