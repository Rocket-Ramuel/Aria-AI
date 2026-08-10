"""Tests for corpus cleaning and the train/validation split.

Both of these bit me during the first real training run: the validation slice
landed entirely on a word-frequency list appended to the end of one source, so
validation loss was measuring the model on something that was not English.
"""

import numpy as np
import pytest

from aria.data import (
    build_chat_examples, clean_plain, clean_wikitext, drop_non_prose,
    encode_dialogue, seed_dialogues, split_train_val,
)
from aria.model import IGNORE_INDEX
from aria.tokenizer import BPETokenizer

PROSE = "\n".join([
    "The river ran past the old mill, turning east towards the sea.",
    "She opened the window and listened to the rain on the roof.",
    "Every morning the baker lit the oven, long before the sky lightened.",
] * 40)

WORD_LIST = "\n".join(["stalwart meats stamping variance apiece firmament"] * 120)


# --- non-prose filtering ---------------------------------------------------


def test_word_lists_are_dropped():
    out = drop_non_prose(PROSE + "\n" + WORD_LIST)
    assert "the old mill" in out
    assert "stalwart meats" not in out


def test_hard_wrapped_prose_survives():
    """Line-wrapped text has punctuation-free *lines* but not blocks; the
    filter must not mistake it for a word list."""
    wrapped = "\n".join(
        "the quick brown fox jumped over the lazy dog and kept running until"
        if i % 4 else "he stopped, breathed, and looked back at the field."
        for i in range(200)
    )
    out = drop_non_prose(wrapped)
    assert len(out) > len(wrapped) * 0.9


def test_filter_keeps_all_of_a_clean_corpus():
    assert drop_non_prose(PROSE) == PROSE


# --- cleaning --------------------------------------------------------------


def test_wikitext_artifacts_are_removed():
    raw = " = Heading = \n The town was founded in 1801 @-@ 1802 . \n <unk> word \n"
    out = clean_wikitext(raw)
    assert "=" not in out
    assert "1801-1802" in out
    assert "<unk>" not in out
    assert " ." not in out


def test_clean_plain_collapses_whitespace_and_blank_lines():
    out = clean_plain("  a   b  \n\n\n  c \t d  \n")
    assert out == "a b\nc d"


# --- train/val split -------------------------------------------------------


def test_split_is_spread_across_the_corpus():
    """The whole point: held-out data must come from everywhere, not just the
    end, or validation measures whichever source was concatenated last."""
    arr = np.arange(200_000, dtype=np.int64)
    train, val = split_train_val(arr, block_size=64, val_frac=0.02)
    # Validation must contain tokens from the first tenth of the corpus...
    assert (val < 20_000).any()
    # ...and from the last tenth.
    assert (val > 180_000).any()


def test_split_is_a_partition():
    arr = np.arange(50_000, dtype=np.int64)
    train, val = split_train_val(arr, block_size=64, val_frac=0.05)
    assert len(train) + len(val) == len(arr)
    assert not (set(train.tolist()) & set(val.tolist())), "train/val overlap"


def test_split_chunks_are_longer_than_the_context_window():
    arr = np.arange(100_000, dtype=np.int64)
    _, val = split_train_val(arr, block_size=256, val_frac=0.01)
    # Contiguity check: a fair LM eval needs runs longer than one context.
    breaks = np.flatnonzero(np.diff(val.astype(np.int64)) != 1)
    runs = np.diff(np.concatenate([[-1], breaks, [len(val) - 1]]))
    assert runs.min() > 256


def test_split_degrades_gracefully_on_a_tiny_corpus():
    arr = np.arange(4000, dtype=np.int64)
    train, val = split_train_val(arr, block_size=64, val_frac=0.05)
    assert len(val) > 64 and len(train) > 0
    assert len(train) + len(val) == len(arr)


# --- chat formatting -------------------------------------------------------


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(PROSE, vocab_size=500)


def test_seed_dialogue_parsing(tmp_path):
    p = tmp_path / "seed.txt"
    p.write_text("# a comment\nU: hello\nA: hi there\n\nU: bye\nA: goodbye\n\nU: orphan\n")
    convos = seed_dialogues(p)
    assert convos == [["hello", "hi there"], ["bye", "goodbye"]]


def test_seed_dialogues_missing_file_is_not_an_error(tmp_path):
    assert seed_dialogues(tmp_path / "nope.txt") == []
    assert seed_dialogues(None) == []


def test_chat_examples_are_built_from_seed_and_corpus(tok, tmp_path):
    import random
    p = tmp_path / "seed.txt"
    p.write_text("U: hello\nA: hello there my friend\n")
    ex = build_chat_examples(tok, PROSE, 64, p, n_surrogate=20, seed_repeat=3,
                             rng=random.Random(0))
    assert len(ex) > 3
    for x, y in ex:
        assert len(x) == len(y)
        assert any(t != IGNORE_INDEX for t in y)


def test_long_dialogue_is_truncated_to_block_size(tok):
    turns = ["a very long user message " * 20, "a very long reply " * 20]
    x, y = encode_dialogue(tok, turns, block_size=64)
    assert len(x) == len(y) == 64


def test_truncation_keeps_arias_reply_not_the_users_preamble(tok):
    """Loss lives only on Aria's turn, which sits at the end. Truncating from
    the head would silently discard the whole training signal."""
    turns = ["background " * 300, "the answer is forty two"]
    x, y = encode_dialogue(tok, turns, block_size=64)
    assert any(t != IGNORE_INDEX for t in y), "reply was truncated away"
    assert x[0] == tok.bos_id
    assert tok.aria_id in x
