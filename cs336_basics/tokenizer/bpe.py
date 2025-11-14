from typing import Optional
import regex as re
import multiprocessing
from tqdm import tqdm
from collections import Counter
from functools import partial

from cs336_basics.pretokenization_example import find_chunk_boundaries


def get_chunks(
    file_path: str,
    desired_num_chunks: int,
) -> list[str]:
    chunks = []
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    
    return chunks

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes] = {}, merges: list[tuple[bytes]] = []):
        self.vocab = vocab
        self.merges = merges

    def _pretokenize(self, text) -> Counter[tuple[bytes]]:
        GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        token_counter = Counter()
        
        for match in re.finditer(GPT2_PAT, text):
            pretoken = match.group() # 'some'
            pretoken_bytes = pretoken.encode("utf-8") # b'some'
            pretoken_tuple = tuple(bytes([b]) for b in pretoken_bytes) # (b's', b'o', b'm', b'e')
            token_counter[pretoken_tuple] += 1

        return token_counter 
    
    def _remove_special_tokens(self, chunk: str, special_tokens: list[str]) -> list[str]:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        pattern = "|".join(escaped_tokens)
        documents = re.split(pattern, chunk)
        
        return documents

    def _find_lex_greatest_pair(self, pair_counter: Counter[tuple[bytes]]) -> tuple[bytes]:
        return max(pair_counter, key=lambda pair: (pair_counter[pair], pair))
        
    def _compute_bpe_merges(self, token_counter: Counter[tuple[bytes]], merges: list[tuple[bytes, bytes]]) -> Optional[tuple[bytes]]:
        pair_counter = Counter()

        for tuple_bytes, count in token_counter.items():
            for i in range(len(tuple_bytes) - 1):
                pair = tuple_bytes[i:i+2]
                pair_counter[pair] += count

        if not pair_counter:
            return None

        merged_pair = self._find_lex_greatest_pair(pair_counter)
        merges.append(merged_pair)

        return merged_pair
    
    def _merge_token(self, token_counter: Counter[tuple[bytes]], merged_pair: tuple[bytes]):
        new_token_counter = Counter() # create a new counter for the merged tokens
        merged_pair_bytes = merged_pair[0] + merged_pair[1] # b's' + b'o' = b'so'

        for tuple_bytes, count in token_counter.items():
            new_tuple_bytes_list: list[bytes] = []
            i = 0

            while i < len(tuple_bytes):
                if i < len(tuple_bytes) - 1 and tuple_bytes[i:i+2] == merged_pair:
                    new_tuple_bytes_list.append(merged_pair_bytes)
                    i += 2
                else:
                    new_tuple_bytes_list.append(tuple_bytes[i])
                    i += 1
            new_token_counter[tuple(new_tuple_bytes_list)] += count
        
        return new_token_counter

    def _process_single_chunk(self, chunk: str, special_tokens: list[str]) -> Counter[tuple[bytes]]:
        documents = self._remove_special_tokens(chunk, special_tokens)
        token_counter = Counter()
        for document in documents:
            doc_token_counter = self._pretokenize(document)
            token_counter.update(doc_token_counter)

        return token_counter

    def train_bpe(self, file_path, vocab_size: int, special_tokens: list[str]):
        self.vocab = {}
        self.merges = []

        # ------------- Step 1. Initialize vocab -------------#
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")
        
        offset = len(special_tokens)
        for i in range(256):
            self.vocab[i + offset] = bytes([i])
        offset += 256

        # ------------- Step 2. Pre-tokenization -------------#
        chunks = get_chunks(
            file_path,
            desired_num_chunks=100,
        )
        token_counter = Counter()

        # parallel processing
        partial_func = partial(self._process_single_chunk, special_tokens=special_tokens)
        num_processes = min(multiprocessing.cpu_count(), len(chunks))
        print(f"Processing {len(chunks)} chunks using {num_processes} processes...")

        with multiprocessing.Pool(processes=num_processes) as pool:
            chunk_counters = list(
                tqdm(
                    pool.imap(partial_func, chunks)
                )
            )
            for chunk_counter in chunk_counters:
                token_counter.update(chunk_counter)

        # ------------- Step 3. Compute Merges -------------#
        with tqdm(total=vocab_size - len(self.vocab)) as pbar:
            i = 0
            while len(self.vocab) < vocab_size:
                merged_pair = self._compute_bpe_merges(token_counter, self.merges)
                if not merged_pair:
                    print("No more merges available. BPE training done.")
                    break
                token_counter = self._merge_token(token_counter, merged_pair)
                self.vocab[offset + i] = merged_pair[0] + merged_pair[1]
                i += 1
                pbar.update(1)
        print(f"{self.merges=}")


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.train_bpe(
        file_path="/data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
