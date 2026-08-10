import pytest

from aria.tokenizer import BPETokenizer, _pretokenize

SAMPLE = (
    "the cat sat on the mat. the cat ate the rat.\n"
    "she sells sea shells by the sea shore, and the shells she sells are sea shells.\n"
) * 40


def test_roundtrip_is_lossless():
    tok = BPETokenizer.train(SAMPLE, vocab_size=600)
    for text in ["the cat sat", "hello, world!", "unseen ünïcödé ✓", "", "   ", "12345"]:
        assert tok.decode(tok.encode(text)) == text


def test_byte_level_covers_arbitrary_input():
    tok = BPETokenizer.train(SAMPLE, vocab_size=400)
    weird = "".join(chr(i) for i in range(32, 500))
    assert tok.decode(tok.encode(weird)) == weird


def test_merges_actually_compress():
    tok = BPETokenizer.train(SAMPLE, vocab_size=800)
    text = "the cat sat on the mat."
    n_bytes = len(text.encode("utf-8"))
    assert len(tok.encode(text)) < n_bytes


def test_special_tokens_are_atomic():
    tok = BPETokenizer.train(SAMPLE, vocab_size=500)
    ids = tok.encode("<|bos|><|user|> hi <|eot|>")
    assert ids[0] == tok.bos_id
    assert ids[1] == tok.user_id
    assert ids[-1] == tok.eot_id
    # and they never appear when specials are disallowed
    assert tok.user_id not in tok.encode("<|user|>", allowed_special=False)


def test_vocab_size_is_respected():
    tok = BPETokenizer.train(SAMPLE, vocab_size=500)
    assert tok.vocab_size <= 500
    assert max(tok.encode("the cat sat")) < tok.vocab_size


def test_too_small_vocab_raises():
    with pytest.raises(ValueError):
        BPETokenizer.train(SAMPLE, vocab_size=10)


def test_save_load(tmp_path):
    tok = BPETokenizer.train(SAMPLE, vocab_size=600)
    p = tmp_path / "tok.json"
    tok.save(p)
    loaded = BPETokenizer.load(p)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("the cat sat on the mat") == tok.encode("the cat sat on the mat")


def test_pretokenizer_keeps_leading_space():
    assert _pretokenize(" hello world") == [" hello", " world"]


@pytest.mark.parametrize("text", [
    "hello world", "  leading and trailing  ", "snake_case_name and __dunder__",
    "a\tb\nc\r\nd", "e=mc^2 & 100% #hash @at", "ünïcödé — em dash … ellipsis",
    "".join(chr(i) for i in range(1, 600)),
])
def test_pretokenizer_is_lossless(text):
    """Every character must land in exactly one piece; anything else silently
    deletes user input on the way into the model."""
    assert "".join(_pretokenize(text)) == text
