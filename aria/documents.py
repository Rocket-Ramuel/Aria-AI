"""Turning uploaded files into something the learner can train on.

Two kinds of upload are worth distinguishing, because they teach different
things:

* **Prose** — an essay, a letter, a diary, notes. It is learned as if Aria had
  written it herself, so her replies drift toward its vocabulary and rhythm.
* **Transcripts** — a chat log or the transcript of someone talking, with
  speaker labels (``Sam: ...``). Given the name of one speaker, that person's
  lines become Aria's side of the conversation and everyone else's become the
  prompts, so she learns how *that person* answers, not just how they write.

Only the standard library is used. `.docx` is a zip of XML and is read
directly; PDF needs the optional `pypdf` package because PDF text extraction
is not something worth reimplementing.
"""

from __future__ import annotations

import io
import re
import zipfile
from xml.etree import ElementTree

TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".text", ".log", ".csv")
SUBTITLE_SUFFIXES = (".srt", ".vtt")
SUPPORTED_SUFFIXES = TEXT_SUFFIXES + SUBTITLE_SUFFIXES + (".docx", ".pdf")

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _docx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as e:
        raise ValueError(f"not a readable .docx file ({e})") from None
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for p in root.iter(f"{_W_NS}p"):
        parts = []
        for node in p.iter():
            if node.tag == f"{_W_NS}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_W_NS}tab":
                parts.append("\t")
            elif node.tag in (f"{_W_NS}br", f"{_W_NS}cr"):
                parts.append("\n")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError(
            "reading PDFs needs the optional pypdf package: pip install pypdf "
            "(or save the document as .txt or .docx)"
        ) from None
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


_SUB_TIMING = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?[.,]\d{3}\s*-->")
_SUB_TAG = re.compile(r"<[^>]+>")


def _subtitle_text(text: str) -> str:
    """Strip cue numbers, timings and markup, leaving only what was said."""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if (not s or s == "WEBVTT" or s.isdigit() or _SUB_TIMING.match(s)
                or s.startswith(("NOTE", "STYLE", "REGION"))):
            continue
        s = _SUB_TAG.sub("", s).strip()
        # Consecutive cues usually split one sentence; repeated cues are common
        # in auto-generated captions.
        if s and (not lines or lines[-1] != s):
            lines.append(s)
    return "\n".join(lines)


def extract_text(name: str, data: bytes) -> str:
    """Plain text from an uploaded file, chosen by its extension."""
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix == ".docx":
        text = _docx_text(data)
    elif suffix == ".pdf":
        text = _pdf_text(data)
    elif suffix in SUBTITLE_SUFFIXES:
        text = _subtitle_text(_decode(data))
    elif suffix in TEXT_SUFFIXES or suffix == "":
        text = _decode(data)
    else:
        raise ValueError(
            f"can't read {suffix} files; use one of {', '.join(SUPPORTED_SUFFIXES)}"
        )
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError(f"{name} contains no text")
    return text


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------

# "Sam: hello", "[10:32] Sam: hello", "Sam Jones - hello" is too ambiguous to
# accept, so only the colon form is recognised.
_SPEAKER_LINE = re.compile(
    r"^\s*(?:\[[^\]]{1,24}\]\s*)?([A-Za-z][\w .'-]{0,30}?)\s*:\s+(\S.*)$"
)


def parse_transcript(text: str, min_share: float = 0.6) -> list[tuple[str, str]] | None:
    """Return [(speaker, utterance), ...] if `text` looks like a transcript.

    A file counts as a transcript when most non-empty lines carry a speaker
    label. Unlabelled lines continue the previous speaker's utterance, which
    is how wrapped messages usually appear in exported chat logs.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    turns: list[tuple[str, str]] = []
    labelled = 0
    for line in lines:
        m = _SPEAKER_LINE.match(line)
        if m:
            labelled += 1
            turns.append((m.group(1).strip(), m.group(2).strip()))
        elif turns:
            speaker, said = turns[-1]
            turns[-1] = (speaker, f"{said} {line.strip()}")
    if labelled / len(lines) < min_share or len({s for s, _ in turns}) < 2:
        return None
    return turns


def speakers(turns: list[tuple[str, str]]) -> list[str]:
    """Speakers in order of how much they said, most first."""
    counts: dict[str, int] = {}
    for s, said in turns:
        counts[s] = counts.get(s, 0) + len(said)
    return sorted(counts, key=counts.get, reverse=True)


def _same(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def transcript_dialogues(
    turns: list[tuple[str, str]], voice: str, context_turns: int = 3
) -> list[list[str]]:
    """Training conversations in which `voice` plays Aria.

    Each of `voice`'s utterances becomes one example: the few turns before it
    as context, alternating user/aria and starting with the user, as
    `encode_dialogue` expects. Utterances with nothing before them to answer
    are skipped, since there is no prompt to learn a response to.
    """
    # Collapse to strictly alternating roles. Everyone who isn't `voice` is
    # "the user"; consecutive lines from the same role are one turn.
    roles: list[list] = []
    for speaker, said in turns:
        role = "aria" if _same(speaker, voice) else "user"
        if roles and roles[-1][0] == role:
            roles[-1][1] += " " + said
        else:
            roles.append([role, said])

    convos: list[list[str]] = []
    for i, (role, _) in enumerate(roles):
        if role != "aria" or i == 0:
            continue
        start = max(0, i - context_turns)
        if roles[start][0] != "user":
            start += 1
        convos.append([said for _, said in roles[start : i + 1]])
    return convos


def strip_speaker_labels(turns: list[tuple[str, str]]) -> str:
    return "\n".join(said for _, said in turns)
