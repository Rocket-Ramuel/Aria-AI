"""Tests for reading uploaded files and turning transcripts into lessons."""

import io
import zipfile

import pytest

from aria.data import encode_dialogue
from aria.documents import (extract_text, parse_transcript, speakers,
                            strip_speaker_labels, transcript_dialogues)
from aria.tokenizer import BPETokenizer


def make_docx(paragraphs):
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paragraphs
    )
    xml = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f"<w:body>{body}</w:body></w:document>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_plain_text_and_markdown():
    assert extract_text("a.txt", b"hello there\r\nfriend") == "hello there\nfriend"
    assert extract_text("notes.md", "# Title\ncafé".encode()) == "# Title\ncafé"


def test_utf16_and_bom_are_handled():
    assert extract_text("a.txt", "﻿hi".encode("utf-8")) == "hi"
    assert extract_text("a.txt", "hello".encode("utf-16")) == "hello"


def test_docx_paragraphs_are_extracted():
    data = make_docx(["First paragraph.", "Second one, with a comma."])
    assert extract_text("letter.docx", data) == "First paragraph.\nSecond one, with a comma."


def test_broken_docx_is_a_clear_error():
    with pytest.raises(ValueError, match="docx"):
        extract_text("bad.docx", b"not a zip")


def test_subtitles_keep_only_the_speech():
    srt = """1
00:00:01,000 --> 00:00:03,000
Well, I reckon <i>that's</i> about right.

2
00:00:03,500 --> 00:00:05,000
Well, I reckon <i>that's</i> about right.

3
00:00:05,000 --> 00:00:07,000
Anyway, how've you been?
"""
    assert extract_text("talk.srt", srt.encode()) == (
        "Well, I reckon that's about right.\nAnyway, how've you been?")
    vtt = "WEBVTT\n\n00:01.000 --> 00:02.000\nhello there\n"
    assert extract_text("talk.vtt", vtt.encode()) == "hello there"


def test_unsupported_and_empty_files_are_rejected():
    with pytest.raises(ValueError, match="can't read"):
        extract_text("song.mp3", b"\x00\x01")
    with pytest.raises(ValueError, match="no text"):
        extract_text("empty.txt", b"   \n ")


def test_pdf_without_pypdf_explains_itself(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_pypdf(name, *a, **kw):
        if name == "pypdf":
            raise ImportError
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_pypdf)
    with pytest.raises(ValueError, match="pip install pypdf"):
        extract_text("doc.pdf", b"%PDF-1.4")


CHAT_LOG = """Sam: hey, are you coming tonight?
Jo: probably not, I'm wiped out
Jo: work was a lot
Sam: fair enough mate
[10:42] Sam: want me to save you a plate
Jo: go on then, cheers
  that's kind of you
"""


def test_transcript_is_detected_and_parsed():
    turns = parse_transcript(CHAT_LOG)
    assert turns is not None
    assert turns[0] == ("Sam", "hey, are you coming tonight?")
    # A continuation line belongs to the previous speaker.
    assert turns[-1] == ("Jo", "go on then, cheers that's kind of you")
    assert set(speakers(turns)) == {"Sam", "Jo"}


def test_prose_is_not_mistaken_for_a_transcript():
    prose = ("The river ran past the mill.\nNote: it was cold that year.\n"
             "Nobody remembered why the mill had closed.\nThe end.")
    assert parse_transcript(prose) is None


def test_transcript_dialogues_make_the_speaker_aria():
    convos = transcript_dialogues(parse_transcript(CHAT_LOG), "jo")   # case-insensitive
    assert convos == [
        ["hey, are you coming tonight?",
         "probably not, I'm wiped out work was a lot"],
        ["hey, are you coming tonight?",
         "probably not, I'm wiped out work was a lot",
         "fair enough mate want me to save you a plate",
         "go on then, cheers that's kind of you"],
    ]
    # Every example starts with the user and ends on the voice being learned,
    # which is the shape encode_dialogue expects.
    for c in convos:
        assert len(c) % 2 == 0


def test_a_speaker_who_opens_the_conversation_is_skipped_for_that_line():
    convos = transcript_dialogues(parse_transcript(CHAT_LOG), "Sam")
    # Sam's opening line answers nobody, so it is not a lesson in replying.
    assert all(c[-1] != "hey, are you coming tonight?" for c in convos)
    assert convos[0][-1] == "fair enough mate want me to save you a plate"


def test_transcript_dialogues_train_only_the_chosen_voice():
    tok = BPETokenizer.train(CHAT_LOG * 20, vocab_size=300)
    convo = transcript_dialogues(parse_transcript(CHAT_LOG), "Jo")[0]
    x, y = encode_dialogue(tok, convo, 128)
    supervised = tok.decode([t for t in y if t >= 0], skip_special=True)
    assert "wiped out" in supervised
    assert "coming tonight" not in supervised


def test_labels_can_be_stripped_for_plain_learning():
    text = strip_speaker_labels(parse_transcript(CHAT_LOG))
    assert "Sam:" not in text and "fair enough mate" in text


def test_stream_decoder_keeps_multibyte_characters_whole():
    tok = BPETokenizer.train("plain ascii words only " * 50, vocab_size=300)
    text = " café \U0001f600 naïve 日本"
    ids = tok.encode(text)
    dec = tok.stream_decoder()
    streamed = "".join(dec.feed(i) for i in ids) + dec.flush()
    assert streamed == text
    # ...which decoding token by token does not manage.
    assert "".join(tok.decode([i]) for i in ids) != text


def test_stream_decoder_skips_special_tokens():
    tok = BPETokenizer.train("hello world " * 50, vocab_size=300)
    dec = tok.stream_decoder()
    out = "".join(dec.feed(i) for i in [tok.bos_id] + tok.encode("hi") + [tok.eot_id])
    assert out == "hi"
