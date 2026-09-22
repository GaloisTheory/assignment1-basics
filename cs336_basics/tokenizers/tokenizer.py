from __future__ import annotations

import pickle
import regex as re

from collections.abc import Iterable, Iterator
from pathlib import Path


from cs336_basics.tokenizers.pretokenization import iter_pretokens


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab

        # Deal with special tokens
        self.merges = merges
        self.inverse_vocab: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        if special_tokens is not None:
            for special_token in special_tokens:
                if special_token.encode("utf-8") in self.inverse_vocab:
                    continue
                else:
                    new_index = len(vocab) - 1
                    while new_index in self.vocab:
                        new_index += 1
                    special_token_bytes = special_token.encode("utf-8")
                    self.vocab[new_index] = special_token_bytes
                    self.inverse_vocab[special_token_bytes] = new_index

            special_tokens.sort(key=len, reverse=True)

        self.special_tokens = special_tokens

        self.merges_index: dict[tuple[bytes, bytes], int] = dict()
        # Create a pairs -> merges dictionary
        for i, pair in enumerate(self.merges):
            if pair in self.merges_index:
                raise ValueError("Can't merge same pair twice")
            self.merges_index[pair] = i

    @classmethod
    def from_files(cls, vocab_filepath: Path, merges_filepath: Path, special_tokens: list | None = None) -> Tokenizer:
        with vocab_filepath.open("rb") as f:
            vocab = pickle.load(f)

        with merges_filepath.open("rb") as f:
            merges = pickle.load(f)

        return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        # Pretokenize
        # Need to find a way to get the special tokens in the tokenizer
        if self.special_tokens is not None:
            st_split_pattern = "|".join(re.escape(s) for s in self.special_tokens)
            chunks = re.split(f"({st_split_pattern})", text)
        else:
            chunks = [text]

        pretokens = set(iter_pretokens(chunks, include_special_tokens=True))
        pretokens_to_tokens: dict[tuple[bytes, ...], list[int]] = dict()
        # Now convert pretokens one at a time
        for pretoken in pretokens:
            # What do we need to do for one pretoken?
            # So we loop until we see a match then we merge it to create a new tuple/list
            # And then we go again, until we finish
            tokenized_pairs = list(pretoken)
            while True:
                index_to_merge = -1
                merge_index = float("inf")
                for i in range(len(tokenized_pairs) - 1):
                    if (tokenized_pairs[i], tokenized_pairs[i + 1]) in self.merges_index:
                        pot_merge_index = self.merges_index[(tokenized_pairs[i], tokenized_pairs[i + 1])]
                        if pot_merge_index < merge_index:
                            index_to_merge = i
                            merge_index = pot_merge_index

                if index_to_merge < 0:
                    pretokens_to_tokens[pretoken] = [self.inverse_vocab[x] for x in tokenized_pairs]
                    break
                else:
                    tokenized_pairs = (
                        tokenized_pairs[:index_to_merge]
                        + [tokenized_pairs[index_to_merge] + tokenized_pairs[index_to_merge + 1]]
                        + tokenized_pairs[index_to_merge + 2 :]
                    )

        tokenization = []
        for pretoken in iter_pretokens(chunks, include_special_tokens=True):
            tokenization += pretokens_to_tokens[pretoken]
        # Ok so now we need the original text again, and then convert it back
        return tokenization

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_joined = b"".join(self.vocab[id] for id in ids)
        return bytes_joined.decode(errors="replace")


if __name__ == "__main__":
    merges = Path(
        "/Users/dohunlee/Documents/001_Projects/011_Stanford_CS336/assignment1-basics/artifacts/tokenizers/TinyStoriesTrain/merges.pkl"
    )
    vocab = Path(
        "/Users/dohunlee/Documents/001_Projects/011_Stanford_CS336/assignment1-basics/artifacts/tokenizers/TinyStoriesTrain/vocab.pkl"
    )
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab, merges_filepath=merges)

    text = "One day, a boy named Tom went to the park with his mom. He liked to run and play on the grass and the swings. He saw many flowers and bugs and birds. He was very happy."

    print(tokenizer.encode(text=text))

    assert tokenizer.decode(tokenizer.encode(text=text)) == text

    print("hello")
