from collections import Counter, defaultdict
from functools import partial
import multiprocessing

import regex as re
from tqdm import tqdm

from cs336_basics.utils.file import stream_chunks_with_split

class BPETokenizer:
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        self.vocab = vocab or {}
        self.merges = merges or []
        self.special_tokens = special_tokens or []

    def pretokenize(self, text: str) -> Counter[bytes]:
        """
        返回 Counter[token_bytes]。大幅减少内存使用。
        """
        GPT2_PAT = (
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        token_counter = Counter()

        for match in re.finditer(GPT2_PAT, text):
            tok = match.group().encode("utf-8")
            token_counter[tok] += 1

        return token_counter

    def _process_single_chunk(self, chunk: bytes, special_tokens: list[str]) -> Counter[bytes]:
        text = chunk.decode("utf-8", errors="ignore")
        for tok in special_tokens:
            text = text.replace(tok, "")
        return self.pretokenize(text)

    def train_bpe(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.vocab = {}
        self.merges = []

        # ---------------- Step 1: init vocab ----------------
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")

        offset = len(special_tokens)

        # 单字节 token
        for i in range(256):
            self.vocab[offset + i] = bytes([i])
        offset += 256

        # ---------------- Step 2: Streaming pretok ----------------
        token_counter = Counter()

        partial_func = partial(self._process_single_chunk, special_tokens=special_tokens)

        print("Streaming large file...")

        # streaming chunks
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for chunk_counter in tqdm(
                pool.imap(partial_func, stream_chunks_with_split(input_path)),
                desc="Pretokenizing"
            ):
                token_counter.update(chunk_counter)

        # -------------------------------------------
        # Convert token_bytes -> list of byte tokens
        # -------------------------------------------
        tokens = []
        token_counts = []

        for token_bytes, count in token_counter.items():
            tokens.append([bytes([b]) for b in token_bytes])  # list[bytes]
            token_counts.append(count)

        del token_counter

        pair_counter = Counter()
        pair_position_table = defaultdict(set)

        for tid, token in enumerate(tokens):
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counter[pair] += token_counts[tid]
                pair_position_table[pair].add(tid)

        # ---------------- Step 4: BPE Merging ----------------
        print("Running BPE merges...")

        for merge_id in tqdm(range(vocab_size - len(self.vocab))):
            if not pair_counter:
                print("No more pairs to merge.")
                break

            # Find most frequent pair (tie-break by lexicographically largest)
            merged_pair = max(pair_counter, key=lambda p: (pair_counter[p], p))
            A, B = merged_pair
            merged_token = A + B

            self.merges.append(merged_pair)
            self.vocab[offset + merge_id] = merged_token

            # Remove old pair
            del pair_counter[merged_pair]

            affected = pair_position_table.pop(merged_pair)
            if not affected:
                continue

            # Apply merge on each affected token
            for tid in affected:
                old = tokens[tid]
                cnt = token_counts[tid]

                # remove old pairs
                for i in range(len(old) - 1):
                    p = (old[i], old[i+1])
                    pair_counter[p] -= cnt

                # construct new merged token
                new_tok = []
                i = 0
                while i < len(old):
                    if i + 1 < len(old) and old[i] == A and old[i+1] == B:
                        new_tok.append(merged_token)
                        i += 2
                    else:
                        new_tok.append(old[i])
                        i += 1

                tokens[tid] = new_tok

            # add new pairs
            for tid in affected:
                new = tokens[tid]
                cnt = token_counts[tid]
                for i in range(len(new) - 1):
                    p = (new[i], new[i + 1])
                    pair_counter[p] += cnt
                    pair_position_table[p].add(tid)

        return self.vocab, self.merges

    def save(self, folder: str):
        """
        Save merges and vocab as GPT-2 format
        """
        import os, json

        os.makedirs(folder, exist_ok=True)

        # 1. Project bytes -> unicode（latin-1 + extension）to ensure the safe json format
        def bytes_to_unicode():
            bs = (
                list(range(33, 127))  # visible ASCII
                + list(range(161, 256))
            )
            cs = bs[:]
            n = 0
            for b in range(256):
                if b not in bs:
                    bs.append(b)
                    cs.append(256 + n)
                    n += 1
            return dict(zip(bs, [chr(c) for c in cs]))

        byte_encoder = bytes_to_unicode()

        def encode_bytes_for_gpt2(bs: bytes) -> str:
            return "".join(byte_encoder[b] for b in bs)

        # 2. save vocab.json
        vocab_dict = {}
        for tok_id, tok_bytes in self.vocab.items():
            token_str = encode_bytes_for_gpt2(tok_bytes)
            vocab_dict[token_str] = tok_id

        with open(f"{folder}/vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

        # 3. save merges.txt
        with open(f"{folder}/merges.txt", "w", encoding="utf-8") as f:
            for a, b in self.merges:
                sa = encode_bytes_for_gpt2(a)
                sb = encode_bytes_for_gpt2(b)
                f.write(f"{sa} {sb}\n")

        print(f"GPT-2 style tokenizer saved to {folder}")


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.train_bpe(
        input_path="/data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.save("/data/tokenizer_owt_32k_optim")