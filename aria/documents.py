"""Turning uploaded files into something the learner can train on.

Two kinds of upload are worth distinguishing, because they teach different
things:

* **Prose** — an essay, a letter, a diary, a book. It is learned as if Aria had
  written it herself: grammar, vocabulary and rhythm all come from it.
* **Transcripts** — a chat log or the transcript of someone talking, with
  speaker labels (``Sam: ...``). Given the name of one speaker, that person's
  lines become Aria's side of the conversation and everyone else's become the
  prompts, so she learns how *that person* answers, not just how they write.

Everything here streams. A document is read line by line, never held in memory
whole, so there is no size limit on what can be uploaded: a 2 GB text file
costs the same memory as a 2 KB one, it just takes longer to learn.

Only the standard library is used. `.docx` is a zip of XML and is parsed
incrementally; PDF needs the optional `pypdf` package because PDF text
extraction is not something worth reimplementing.
"""

from __future__ import annotations

import codecs
import io
import re
import zipfile
from collections import deque
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Union
from xml.etree import ElementTree

TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".text", ".log", ".csv")
SUBTITLE_SUFFIXES = (".srt", ".vtt")
SUPPORTED_SUFFIXES = TEXT_SUFFIXES + SUBTITLE_SUFFIXES + (".docx", ".pdf")

# A line longer than this is handed on in pieces, so a file with no newlines
# at all can't make the reader hold the whole thing.
MAX_LINE_CHARS = 16_384
_READ_CHUNK = 1 << 18

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

Source = Union[str, Path, bytes, BinaryIO]


def suffix_of(name: str) -> str:
    return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _open(source: Source) -> BinaryIO:
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    if isinstance(source, (str, Path)):
        return open(source, "rb")
    return source


def _pick_encoding(head: bytes) -> str:
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    try:
        # final=False: the sample may end in the middle of a character.
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def _split_long(line: str) -> Iterator[str]:
    while len(line) > MAX_LINE_CHARS:
        cut = line.rfind(" ", 0, MAX_LINE_CHARS)
        cut = cut if cut > MAX_LINE_CHARS // 2 else MAX_LINE_CHARS
        yield line[:cut]
        line = line[cut:].lstrip()
    yield line


def _iter_decoded_lines(f: BinaryIO) -> Iterator[str]:
    head = f.read(1 << 16)
    enc = _pick_encoding(head)
    dec = codecs.getincrementaldecoder(enc)(errors="replace")
    pending = ""
    chunk = head
    while chunk:
        pending += dec.decode(chunk)
        *lines, pending = pending.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for line in lines:
            yield from _split_long(line)
        if len(pending) > MAX_LINE_CHARS:
            *pieces, pending = list(_split_long(pending))
            yield from pieces
        chunk = f.read(_READ_CHUNK)
    pending += dec.decode(b"", final=True)
    if pending:
        yield from _split_long(pending)


_SUB_TIMING = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?[.,]\d{3}\s*-->")
_SUB_TAG = re.compile(r"<[^>]+>")


def _subtitle_lines(lines: Iterable[str]) -> Iterator[str]:
    """Strip cue numbers, timings and markup, leaving only what was said."""
    last = None
    for line in lines:
        s = line.strip()
        if (not s or s == "WEBVTT" or s.isdigit() or _SUB_TIMING.match(s)
                or s.startswith(("NOTE", "STYLE", "REGION"))):
            continue
        s = _SUB_TAG.sub("", s).strip()
        # Auto-generated captions repeat a cue across several timings.
        if s and s != last:
            yield s
            last = s


def _docx_lines(f: BinaryIO) -> Iterator[str]:
    try:
        z = zipfile.ZipFile(f)
        member = z.open("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as e:
        raise ValueError(f"not a readable .docx file ({e})") from None
    body = None
    with member:
        for event, elem in ElementTree.iterparse(member, events=("start", "end")):
            if event == "start":
                if elem.tag == f"{_W_NS}body":
                    body = elem
                continue
            if elem.tag != f"{_W_NS}p":
                continue
            parts = []
            for node in elem.iter():
                if node.tag == f"{_W_NS}t" and node.text:
                    parts.append(node.text)
                elif node.tag == f"{_W_NS}tab":
                    parts.append("\t")
                elif node.tag in (f"{_W_NS}br", f"{_W_NS}cr"):
                    parts.append("\n")
            for line in "".join(parts).split("\n"):
                yield from _split_long(line)
            # Drop what has been read, so a long document is never held whole.
            elem.clear()
            if body is not None:
                body.clear()


def _pdf_lines(f: BinaryIO) -> Iterator[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError(
            "reading PDFs needs the optional pypdf package: pip install pypdf "
            "(or save the document as .txt or .docx)"
        ) from None
    for page in PdfReader(f).pages:
        for line in (page.extract_text() or "").split("\n"):
            yield from _split_long(line)
        yield ""


def iter_lines(source: Source, name: str) -> Iterator[str]:
    """Stream a document's text as lines, choosing the reader by extension."""
    suffix = suffix_of(name)
    if suffix not in SUPPORTED_SUFFIXES and suffix != "":
        raise ValueError(
            f"can't read {suffix} files; use one of {', '.join(SUPPORTED_SUFFIXES)}"
        )
    f = _open(source)
    try:
        if suffix == ".docx":
            yield from _docx_lines(f)
        elif suffix == ".pdf":
            yield from _pdf_lines(f)
        elif suffix in SUBTITLE_SUFFIXES:
            yield from _subtitle_lines(_iter_decoded_lines(f))
        else:
            yield from _iter_decoded_lines(f)
    finally:
        if f is not source:
            f.close()


def extract_text(name: str, data: Source) -> str:
    """The whole text of a (small) document. Uploads use `iter_lines`."""
    text = "\n".join(iter_lines(data, name)).strip()
    if not text:
        raise ValueError(f"{name} contains no text")
    return text


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def iter_units(lines: Iterable[str]) -> Iterator[str]:
    """Sentences, more or less: the pieces documents are learned in."""
    for line in lines:
        for u in _SENTENCE_END.split(line):
            u = u.strip()
            if len(u) > 1:
                yield u


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------

# "Sam: hello", "[10:32] Sam: hello". "Sam Jones - hello" is too ambiguous to
# accept, so only the colon form is recognised.
_SPEAKER_LINE = re.compile(
    r"^\s*(?:\[[^\]]{1,24}\]\s*)?([A-Za-z][\w .'-]{0,30}?)\s*:\s+(\S.*)$"
)
# One merged turn is capped so a mis-detected file can't build an unbounded
# string; only the end of an over-long turn could fit in context anyway.
MAX_TURN_CHARS = 8_000


def iter_turns(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """(speaker, utterance) pairs. Unlabelled lines continue the previous
    speaker's utterance, which is how wrapped messages appear in exported
    chat logs; lines before the first label are dropped."""
    current: list | None = None
    for line in lines:
        if not line.strip():
            continue
        m = _SPEAKER_LINE.match(line)
        if m:
            if current is not None:
                yield current[0], current[1]
            current = [m.group(1).strip(), m.group(2).strip()]
        elif current is not None:
            current[1] = (current[1] + " " + line.strip())[-MAX_TURN_CHARS:]
    if current is not None:
        yield current[0], current[1]


def parse_transcript(text_or_lines: str | Iterable[str],
                     min_share: float = 0.6) -> list[tuple[str, str]] | None:
    """Return [(speaker, utterance), ...] if the text looks like a transcript.

    A file counts as a transcript when most non-empty lines carry a speaker
    label. Pass the first few hundred lines of a large file to sniff it.
    """
    if isinstance(text_or_lines, str):
        text_or_lines = text_or_lines.splitlines()
    lines = [l for l in text_or_lines if l.strip()]
    if len(lines) < 2:
        return None
    labelled = sum(1 for l in lines if _SPEAKER_LINE.match(l))
    turns = list(iter_turns(lines))
    if labelled / len(lines) < min_share or len({s for s, _ in turns}) < 2:
        return None
    return turns


def speakers(turns: Iterable[tuple[str, str]]) -> list[str]:
    """Speakers in order of how much they said, most first."""
    counts: dict[str, int] = {}
    for s, said in turns:
        counts[s] = counts.get(s, 0) + len(said)
    return sorted(counts, key=counts.get, reverse=True)


def _same(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def iter_dialogues(
    turns: Iterable[tuple[str, str]], voice: str, context_turns: int = 3
) -> Iterator[list[str]]:
    """Training conversations in which `voice` plays Aria, streamed.

    Each of `voice`'s utterances becomes one example: the few turns before it
    as context, alternating user/aria and starting with the user, as
    `encode_dialogue` expects. Utterances with nothing before them to answer
    are skipped, since there is no prompt to learn a response to.
    """
    # Strictly alternating roles: everyone who isn't `voice` is "the user",
    # and consecutive lines from the same role are one turn. A turn is only
    # complete once the other role speaks, so emission lags by one turn.
    window: deque[list] = deque(maxlen=context_turns + 1)
    seen_any = False

    def emit():
        roles = list(window)
        if roles[-1][0] != "aria" or len(roles) < 2:
            return None
        start = 0 if roles[0][0] == "user" else 1
        return [said for _, said in roles[start:]]

    for speaker, said in turns:
        role = "aria" if _same(speaker, voice) else "user"
        if window and window[-1][0] == role:
            window[-1][1] = (window[-1][1] + " " + said)[-MAX_TURN_CHARS:]
            continue
        if window:
            convo = emit()
            if convo:
                yield convo
        window.append([role, said])
        seen_any = True
    if seen_any:
        convo = emit()
        if convo:
            yield convo


def transcript_dialogues(turns: list[tuple[str, str]], voice: str,
                         context_turns: int = 3) -> list[list[str]]:
    return list(iter_dialogues(turns, voice, context_turns))


def strip_speaker_labels(turns: Iterable[tuple[str, str]]) -> str:
    return "\n".join(said for _, said in turns)
