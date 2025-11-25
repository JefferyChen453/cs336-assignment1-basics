import multiprocessing
from multiprocessing import Pool
import os
from typing import Tuple

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.utils.file import stream_chunks_with_split

def process_chunk(args: Tuple):
    tokenizer, chunk_bytes, output_folder, idx = args
    try:
        # decode bytes -> str
        text = chunk_bytes.decode("utf-8", errors="ignore")

        # encode to token ids
        ids = tokenizer.encode(text)

        # serialize as uint16
        id_array = np.array(ids, dtype=np.uint16)

        # write file
        output_path = os.path.join(output_folder, f"{idx:05d}.bin")
        id_array.tofile(output_path)
    except Exception as e:
        print(f"[Error] chunk {idx}: {e}")
        raise

def tokenize_parallel(
    tokenizer,
    input_path: str,
    output_folder: str,
    num_workers: int = 8,
):
    os.makedirs(output_folder, exist_ok=True)

    # get total size for progress bar
    block_size = 256 * 1024 * 1024

    # pool for parallel encoding
    pool = Pool(num_workers)
    futures = []

    for idx, chunk in enumerate(
        tqdm(
            stream_chunks_with_split(
                input_path,
                block_size=block_size,
                reserve_split_token=True
            ),
            desc="Tokenizing chunks..."
        )
    ):
        futures.append(
            pool.apply_async(
                process_chunk,
                args=((tokenizer, chunk, output_folder, idx),)
            )
        )

    # wait for finish
    for f in futures:
        f.get()

    pool.close()
    pool.join()


def main():
    num_processes = multiprocessing.cpu_count()

    tinystories_tokenizer = Tokenizer.from_files(
        "/data/tokenizer_TinyStories_10k/vocab.json",
        "/data/tokenizer_TinyStories_10k/merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    tokenize_parallel(tinystories_tokenizer, "/data/TinyStoriesV2-GPT4-train.txt", "/data/data_bin/tinystories_train", num_processes)
    tokenize_parallel(tinystories_tokenizer, "/data/TinyStoriesV2-GPT4-valid.txt", "/data/data_bin/tinystories_valid", num_processes)

    owt_tokenizer = Tokenizer.from_files(
        "/data/tokenizer_owt_32k_optim/vocab.json",
        "/data/tokenizer_owt_32k_optim/merges.txt",
        special_tokens=["<|endoftext|>"],
    )

    tokenize_parallel(owt_tokenizer, "/data/owt_train.txt", "/data/data_bin/owt_train", num_processes)
    tokenize_parallel(owt_tokenizer, "/data/owt_valid.txt", "/data/data_bin/owt_valid", num_processes)


if __name__ == "__main__":
    main()