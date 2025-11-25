import random
import time

import regex as re

from cs336_basics.tokenizer.bpe_optim import BPETokenizer
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.utils.profiling import profile_time_memory_gpu


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def find_longest_token(vocab: dict[int, bytes]):
    longest_token = max(vocab.values(), key=len)
    return longest_token.decode("utf-8", errors="ignore")

def sample_documents(input_path, sample_num = 10, block_size= 32 * 1024 * 1024):
    with open(input_path, "rb") as f:
        block = f.read(block_size)
        pattern = re.compile(re.escape(b"<|endoftext|>"))
        matches = pattern.split(block)[:-1]

        sample_docs = random.sample(matches, k=sample_num)
        sample_docs = [doc.decode("utf-8", errors="ignore") for doc in sample_docs]
        
        return "".join(sample_docs)

@profile_time_memory_gpu()
@print_func_name()
def train_bpe_tinystories():
    tokenizer = BPETokenizer()
    vocab, merges = tokenizer.train_bpe(
        input_path="/data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    longest_token = find_longest_token(vocab)
    print("Longest token in TinyStories:", longest_token)

@profile_time_memory_gpu()
@print_func_name()
def train_bpe_expts_owt():
    tokenizer = BPETokenizer()
    vocab, merges = tokenizer.train_bpe(
        input_path="/data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    longest_token = find_longest_token(vocab)
    print("Longest token in OpenWebText:", longest_token)

@print_func_name()
def tokenizer_experiments():
    sampled_tinystories = sample_documents("/data/TinyStoriesV2-GPT4-train.txt")
    sampled_owt = sample_documents("/data/owt_train.txt")
    tinystories_tokenizer = Tokenizer.from_files(
        vocab_filepath="/data/tokenizer_TinyStories_10k_optim/vocab.json",
        merges_filepath="/data/tokenizer_TinyStories_10k_optim/merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath="/data/tokenizer_owt_32k_optim/vocab.json",
        merges_filepath="/data/tokenizer_owt_32k_optim/merges.txt",
        special_tokens=["<|endoftext|>"]
    )

    # Problem (a)
    print("-" * 50, "Problem (a)", "-" * 50)
    sampled_tinystories_ids = tinystories_tokenizer.encode(sampled_tinystories)
    sampled_owt_ids = owt_tokenizer.encode(sampled_owt)
    print("TinyStories compression rate:", len(sampled_tinystories) / len(sampled_tinystories_ids))
    print("OpenWebText compression rate:", len(sampled_owt) / len(sampled_owt_ids))

    # Problem (b)
    print("-" * 50, "Problem (b)", "-" * 50)
    ids = tinystories_tokenizer.encode(sampled_owt)
    print("Tokenize OpenWebText with TinyStories Tokenizer:", len(sampled_owt) / len(ids))
    
    # Problem (c)
    print("-" * 50, "Problem (c)", "-" * 50)
    start_time = time.time()
    text = sample_documents("/data/owt_train.txt", sample_num=1000, block_size=128 * 1024 * 1024)
    ids = owt_tokenizer.encode(text)
    end_time = time.time()
    throughput = len(text.encode("utf-8")) / (end_time - start_time)
    print(f"Throughput: {throughput / (1024 ** 2)} MB/second")
    print("Encoding Pile dataset takes:", 825 * (1024 ** 3) / throughput / 3600, "hours")
    print('-' * 100)

    # Problem (d)
    # see tokenize_corpus.py


def main():
    # train_bpe_tinystories()
    # train_bpe_expts_owt()
    tokenizer_experiments()

if __name__ == "__main__":
    main()