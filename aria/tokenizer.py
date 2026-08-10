"""A byte-level BPE tokenizer, trained from scratch on the target corpus.

No pretrained vocabulary is downloaded or reused: `BPETokenizer.train` learns
the merge table from raw bytes. Byte-level means every possible input string is
encodable, so the model can never hit an unknown token.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from .config import SPECIAL_TOKENS

# GPT-2's pre-tokenization pattern. Splitting here stops BPE from ever merging
# across whitespace or between a word and its punctuation, which keeps the
# learned vocabulary interpretable and the merge search tractable.
#
# `[^\W\d_]` is Python's stand-in for `\p{L}` (letters only), and the symbol
# class has to re-admit `_` because `\w` counts it as a word character. The
# trailing `\S|\s` catch-all guarantees the split is lossless: every character
# of the input lands in exactly one piece, so encode/decode always round-trips.
_SPLIT_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+|\S"""
)


def _pretokenize(text: str) -> list[str]:
    return _SPLIT_PAT.findall(text)


class BPETokenizer:
    """Byte-level BPE.

    Token ids are laid out as::

        [0 .. len(specials))            special tokens
        [len(specials) .. +256)         the 256 raw bytes
        [.. vocab_size)                 learned merges, in merge order
    """

    def __init__(
        self,
        merges: Sequence[tuple[int, int]] | None = None,
        specials: Sequence[str] = SPECIAL_TOKENS,
    ) -> None:
        self.specials = list(specials)
        self.n_special = len(self.specials)
        self.special_to_id = {tok: i for i, tok in enumerate(self.specials)}
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._rebuild()

    # -- construction -------------------------------------------------------

    def _rebuild(self) -> None:
        self.byte_offset = self.n_special
        self.merge_offset = self.n_special + 256
        # rank[(a, b)] -> new token id
        self.rank: dict[tuple[int, int], int] = {
            pair: self.merge_offset + i for i, pair in enumerate(self.merges)
        }
        # token id -> byte string, for decoding
        self.token_bytes: list[bytes] = [b""] * self.n_special
        self.token_bytes += [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return self.merge_offset + len(self.merges)

    # ids of tokens that are convenient to have to hand
    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eot_id(self) -> int:
        return self.special_to_id["<|eot|>"]

    @property
    def user_id(self) -> int:
        return self.special_to_id["<|user|>"]

    @property
    def aria_id(self) -> int:
        return self.special_to_id["<|aria|>"]

    # -- training -----------------------------------------------------------

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        specials: Sequence[str] = SPECIAL_TOKENS,
        min_freq: int = 2,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Learn a merge table from `text`.

        Words are collapsed to (symbol-sequence, frequency) pairs first, so the
        merge loop costs O(unique words) rather than O(corpus bytes). Pair counts
        are maintained incrementally: applying a merge only touches the words
        that actually contain the merged pair.
        """
        n_special = len(specials)
        byte_offset = n_special
        n_merges = vocab_size - n_special - 256
        if n_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} is too small; need at least {n_special + 256}"
            )

        freqs: dict[str, int] = defaultdict(int)
        for piece in _pretokenize(text):
            freqs[piece] += 1

        # words[i] is a mutable list of symbol ids; counts[i] its corpus frequency
        words: list[list[int]] = []
        counts: list[int] = []
        for piece, freq in freqs.items():
            if freq < min_freq:
                continue
            words.append([byte_offset + b for b in piece.encode("utf-8")])
            counts.append(freq)

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for idx, word in enumerate(words):
            c = counts[idx]
            for pair in zip(word, word[1:]):
                pair_counts[pair] += c
                pair_where[pair].add(idx)

        merges: list[tuple[int, int]] = []
        next_id = n_special + 256

        for step in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] < min_freq:
                break

            merges.append(best)
            new_id = next_id
            next_id += 1

            affected = list(pair_where[best])
            for idx in affected:
                word = words[idx]
                c = counts[idx]

                # Remove this word's contribution to every pair it currently has.
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= c
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                    where = pair_where.get(pair)
                    if where is not None:
                        where.discard(idx)

                # Rewrite the word with `best` collapsed into `new_id`.
                merged: list[int] = []
                i = 0
                a, b = best
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                words[idx] = merged

                # Add the rewritten word's pairs back in.
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += c
                    pair_where[pair].add(idx)

            pair_counts.pop(best, None)
            pair_where.pop(best, None)

            if verbose and (step + 1) % 500 == 0:
                print(f"  bpe merge {step + 1}/{n_merges}", flush=True)

        return cls(merges=merges, specials=specials)

    # -- encoding -----------------------------------------------------------

    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        ids = [self.byte_offset + b for b in piece.encode("utf-8")]
        while len(ids) >= 2:
            # Greedily apply the earliest-learned merge present in the sequence.
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pos = r, i
            if best_rank is None:
                break
            ids[best_pos : best_pos + 2] = [best_rank]

        if len(self._cache) < 200_000:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode text to ids. Special-token literals in the text are honoured
        when `allowed_special` is set, which is how the chat format is built."""
        if not allowed_special or not self.specials:
            return [i for p in _pretokenize(text) for i in self._encode_piece(p)]

        pattern = "(" + "|".join(re.escape(s) for s in self.specials) + ")"
        out: list[int] = []
        for chunk in re.split(pattern, text):
            if not chunk:
                continue
            if chunk in self.special_to_id:
                out.append(self.special_to_id[chunk])
            else:
                for p in _pretokenize(chunk):
                    out.extend(self._encode_piece(p))
        return out

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        buf = bytearray()
        for i in ids:
            if i < self.n_special:
                if not skip_special:
                    buf.extend(self.specials[i].encode("utf-8"))
                continue
            buf.extend(self.token_bytes[i])
        return buf.decode("utf-8", errors="replace")

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"specials": self.specials, "merges": self.merges})
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text())
        return cls(merges=[tuple(m) for m in d["merges"]], specials=d["specials"])
