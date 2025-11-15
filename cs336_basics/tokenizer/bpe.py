from typing import Optional
import regex as re
import multiprocessing
import json
from tqdm import tqdm
from collections import Counter, defaultdict
from functools import partial

from cs336_basics.pretokenization_example import get_chunks

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes] = {},
        merges: list[tuple[bytes]] = [],
        special_tokens: list[str] = []
    ):
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
        
    def _merge_one_pair(
        self,
        merged_pair: tuple[bytes, bytes],
        tokens: list[list[bytes]],
        token_counts: list[int],
        pair_counter: Counter[tuple[bytes]],
        pair_position_table: dict[tuple[bytes], set[int]]
    ) -> Optional[tuple[bytes]]:
        A, B = merged_pair
        merged_pair_bytes = A + B

        affected_token_ids = list(pair_position_table[merged_pair])

        for token_idx in affected_token_ids:
            old_token = tokens[token_idx]
            new_token = []
            i = 0
            while i < len(old_token):
                if i + 1 < len(old_token) and (old_token[i], old_token[i + 1]) == merged_pair:
                    new_token.append(merged_pair_bytes)
                    i += 2
                else:
                    new_token.append(old_token[i])
                    i += 1
            tokens[token_idx] = new_token
        
        for pair in list(pair_position_table.keys()): # use list to save a copy of dict
            # filter out the pairs that are not affected by the merge
            pair_position_table[pair].difference_update(affected_token_ids)
        
        # recompute the pair count&position of affected tokens
        for token_idx in affected_token_ids:
            token = tokens[token_idx]
            token_count = token_counts[token_idx]
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counter[pair] += token_count
                pair_position_table[pair].add(token_idx)


    def _process_single_chunk(self, chunk: str, special_tokens: list[str]) -> Counter[tuple[bytes]]:
        documents = self._remove_special_tokens(chunk, special_tokens)
        token_counter = Counter()
        for document in documents:
            doc_token_counter = self._pretokenize(document)
            token_counter.update(doc_token_counter)

        return token_counter

    def train_bpe(self, input_path, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
            input_path,
            desired_num_chunks=1000,
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

        # convert token_counter to tokens & token_counts
        tokens: list[list[bytes]] = []
        token_counts: list[int] = []
        for tuple_bytes, count in token_counter.items():
            tokens.append(list(tuple_bytes))
            token_counts.append(count)
        del token_counter

        # ------------- Step 3. Compute Merges -------------#
        # Count pairs
        pair_counter = Counter()
        pair_position_table = defaultdict(set) # record the positions of each pair in the token_counter eg. pair_position_table[(b's', b'o')] = [(0, 2), (2, 3)] means: b'some' has b'so' at 2 in word0 and 3 in word 2
        for token_idx, token in enumerate(tokens):
            for pair_pos in range(len(token) - 1):
                pair = (token[pair_pos], token[pair_pos+1])
                pair_counter[pair] += token_counts[token_idx]
                pair_position_table[pair].add(token_idx)

        # BPE process
        with tqdm(total=vocab_size - len(self.vocab)) as pbar:
            i = 0
            while len(self.vocab) < vocab_size:
                if not pair_counter:
                    print("No more pairs to merge!")
                    break
                merged_pair = self._find_lex_greatest_pair(pair_counter)
                self.merges.append(merged_pair)
                self.vocab[offset + i] = merged_pair[0] + merged_pair[1]

                del pair_counter[merged_pair]

                self._merge_one_pair(merged_pair, tokens, token_counts, pair_counter, pair_position_table)

                i += 1
                pbar.update(1)

        return self.vocab, self.merges

    def save(self, folder: str):
        import os
        os.makedirs(li, exist_ok=True)

        # save vocab
        vocab_dict = {str(i): v.hex() for i, v in self.vocab.items()}
        with open(f"{folder}/vocab.json", "w") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

        # save merges
        with open(f"{folder}/merges.txt", "w") as f:
            for a, b in self.merges:
                f.write(f"{a.hex()} {b.hex()}\n")

        print(f"Tokenizer saved to {folder}")

    @classmethod
    def load(cls, folder: str):
        tokenizer = cls()

        # load vocab
        with open(f"{folder}/vocab.json") as f:
            vocab_json = json.load(f)
            tokenizer.vocab = {int(k): bytes.fromhex(v) for k, v in vocab_json.items()}

        # load merges
        merges = []
        with open(f"{folder}/merges.txt") as f:
            for line in f:
                a_hex, b_hex = line.strip().split()
                merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))
        tokenizer.merges = merges
        
        print(f"Tokenizer loaded from {folder}")
        return tokenizer

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.train_bpe(
        input_path="/data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.save("/data/my_bpe_tokenizer_owt")