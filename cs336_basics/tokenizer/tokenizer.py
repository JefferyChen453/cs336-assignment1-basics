from typing import Iterable, Iterator
import re
import regex


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self._split_pattern = None

        # Build a lookup from byte sequence to token id
        self.byte_to_id = {v: k for k, v in self.vocab.items()}

        # Encode special tokens into bytes
        self.special_tokens_bytes = [s.encode("utf-8") for s in self.special_tokens]
        self.special_tokens_set = set(self.special_tokens_bytes)

        for token in self.special_tokens_bytes:
            if token not in self.byte_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token
                self.byte_to_id[token] = new_id

        # Prepare for efficient merge operations
        self.merges = [(a, b) for a, b in merges]
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Pre-compile regex patterns
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern_str = "|".join(re.escape(tok) for tok in sorted_specials)
            self._split_pattern = re.compile(f"({pattern_str})")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "BPETokenizer":
        import json
        import os

        # ---------- 1. Recreate GPT-2 byte → unicode mapping ----------
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
        # reverse mapping
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        def decode_gpt2_str_to_bytes(s: str) -> bytes:
            """Convert GPT-2 unicode-extended token string back to bytes."""
            return bytes([byte_decoder[ch] for ch in s])

        # ---------- 2. Load vocab.json ----------
        if not os.path.exists(vocab_filepath):
            raise FileNotFoundError(f"Vocab file not found: {vocab_filepath}")

        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_data = json.load(vf)  # {token_str: id}

        vocab: dict[int, bytes] = {}
        for tok_str, tok_id in vocab_data.items():
            tok_bytes = decode_gpt2_str_to_bytes(tok_str)
            vocab[int(tok_id)] = tok_bytes

        # ---------- 3. Load merges.txt ----------
        merges = []
        if os.path.exists(merges_filepath):
            with open(merges_filepath, "r", encoding="utf-8") as mf:
                for line in mf:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    a = decode_gpt2_str_to_bytes(parts[0])
                    b = decode_gpt2_str_to_bytes(parts[1])
                    merges.append((a, b))

        # ---------- 4. Return Tokenizer ----------
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens or [])

    def _byte_pair_merge(self, token: bytes) -> list[bytes]:
        # Convert bytes to tuple of single-byte elements
        word = [bytes([b]) for b in token]
        pairs = lambda w: set((w[i], w[i + 1]) for i in range(len(w) - 1))

        while True:
            candidate_pairs = pairs(word)
            ranked_pairs = [(self.merge_ranks[p], p) for p in candidate_pairs if p in self.merge_ranks]
            if not ranked_pairs:
                break

            _, best_pair = min(ranked_pairs)
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text: str) -> list[int]:
        PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        result = []
        segments = re.split(self._split_pattern, text) if self._split_pattern else [text]

        for segment in segments:
            if not segment:
                continue
            b = segment.encode("utf-8")
            if b in self.special_tokens_set:
                result.append(self.byte_to_id[b])
            else:
                for match in regex.finditer(PAT, segment):
                    token = match.group()
                    for merged in self._byte_pair_merge(token.encode("utf-8")):
                        result.append(self.byte_to_id[merged])
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")


if __name__ == "__main__":
    text = "hello ! ！，4#"
    tokenizer = Tokenizer.from_files('/data/tokenizer_TinyStories_10k/vocab.json', '/data/tokenizer_TinyStories_10k/merges.txt', ["<|endoftext|>"])
    encoded_text = tokenizer.encode(text)
    print(encoded_text)
    for token_id in encoded_text:
        print(f"{token_id}: {tokenizer.vocab[token_id]}")
    print(tokenizer.decode(encoded_text))
