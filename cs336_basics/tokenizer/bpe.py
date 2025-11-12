from typing import Any
import regex as re
import multiprocessing
from collections import Counter

from ..pretokenization_example import find_chunk_boundaries



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
        segments = re.split(pattern, chunk)
        
        return segments

    def _find_lex_greatest_pair(self, pair_counter: Counter[tuple[bytes]]) -> tuple[bytes]:
        return max(pair_counter, key=lambda pair: (pair_counter[pair], pair))
        
    def _compute_bpe_merges(self, token_counter: Counter[tuple[bytes]]):
        merges = []
        pair_counter = Counter()

        for tuple_bytes, count in token_counter.items():
            for i in range(len(tuple_bytes) - 1):
                pair = tuple_bytes[i:i+2]
                pair_counter[pair] += count
        merged_pair = self._find_lex_greatest_pair(pair_counter)
        merges.append(merged_pair)
    
    def _merge_token(self, token_counter: Counter[tuple[bytes]], merged_pair: tuple[bytes]):
        new_token_counter = Counter() # create a new counter for the merged tokens
        merged_pair_bytes = merged_pair[0] + merged_pair[1] # b's' + b'o' = b'so'

        for tuple_bytes, count in token_counter.items():
            new_tuple_bytes_list = []
            i = 0
            while i < len(tuple_bytes) - 1:
                if tuple_bytes[i:i+2] == merged_pair_bytes:
                    new_tuple_bytes_list.append(merged_pair_bytes)
                    i += 2
                else:
                    new_tuple_bytes_list.append(tuple_bytes[i])
                    i += 1
            new_token_counter[tuple(new_tuple_bytes_list)] += count
        
        return new_token_counter

    def train_bpe(self, file_path, special_tokens: list[str]):
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
        

