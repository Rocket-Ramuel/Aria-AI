"""Corpus acquisition, cleaning, tokenization and batching.

`prepare()` turns raw English text into three artifacts under `data/`:

    tokenizer.json     the BPE merge table learned from this corpus
    train.bin/val.bin  uint16 token streams for language-model pretraining
    chat.pt            turn-structured examples with a loss mask, so the model
                       learns the <|user|>/<|aria|> protocol rather than just
                       free-running prose
"""

from __future__ import annotations

import gzip
import json
import pickle
import random
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from .model import IGNORE_INDEX
from .tokenizer import BPETokenizer

# Public-domain / permissively-hosted plain English text. Both are small enough
# to fetch in seconds and are prose rather than markup.
SOURCES: dict[str, str] = {
    "wikitext2_train.txt": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "wikitext2_valid.txt": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "wikitext2_test.txt": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
    "big.txt": "https://raw.githubusercontent.com/dscape/spell/master/test/resources/big.txt",
}

_WIKI_HEADING = re.compile(r"^\s*=+ .* =+\s*$")
_WS = re.compile(r"[ \t]+")


def download(data_dir: Path, sources: dict[str, str] = SOURCES) -> list[Path]:
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in sources.items():
        dest = raw / name
        if dest.exists() and dest.stat().st_size > 0:
            paths.append(dest)
            continue
        print(f"  fetching {name} ...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  ! skipping {name}: {e}")
            continue
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        dest.write_bytes(data)
        paths.append(dest)
    if not paths:
        raise RuntimeError(
            "no corpus could be downloaded; place plain .txt files in data/raw/ "
            "and re-run with --offline"
        )
    return paths


def clean_wikitext(text: str) -> str:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or _WIKI_HEADING.match(line):
            continue
        # wikitext-103 tokenisation artifacts
        line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        line = line.replace(" <unk>", "").replace("<unk>", "")
        line = re.sub(r"\s+([,.!?;:'])", r"\1", line)
        line = re.sub(r"\(\s+", "(", line)
        line = re.sub(r"\s+\)", ")", line)
        out.append(_WS.sub(" ", line))
    return "\n".join(out)


def clean_plain(text: str) -> str:
    out = []
    for line in text.split("\n"):
        line = _WS.sub(" ", line.strip())
        if line:
            out.append(line)
    return "\n".join(out)


def drop_non_prose(text: str, block_lines: int = 60,
                   min_punct_per_1k: float = 3.0) -> str:
    """Remove blocks that are not running prose.

    Public-domain text dumps tend to carry appendices that are not sentences —
    word-frequency lists, indices, tables, code. `big.txt` ends with ~145 KB of
    one-word-per-line vocabulary, which teaches a small model nothing and
    poisons any validation slice that lands on it.

    The test is sentence-punctuation density measured over a block of lines.
    Real prose runs around 20 marks per 1000 characters even when hard-wrapped,
    so a threshold of 3 discards word lists without touching anything a person
    would call text.
    """
    lines = text.split("\n")
    kept: list[str] = []
    for i in range(0, len(lines), block_lines):
        block = lines[i : i + block_lines]
        joined = "\n".join(block)
        if not joined.strip():
            continue
        density = 1000 * sum(joined.count(c) for c in ".,!?;:") / max(1, len(joined))
        if density >= min_punct_per_1k:
            kept.extend(block)
    return "\n".join(kept)


def build_corpus(paths: Sequence[Path], filter_non_prose: bool = True) -> str:
    chunks = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        text = clean_wikitext(text) if "wikitext" in p.name else clean_plain(text)
        if filter_non_prose:
            text = drop_non_prose(text)
        chunks.append(text)
    return "\n".join(chunks)


def split_train_val(arr: np.ndarray, block_size: int, val_frac: float,
                    n_chunks: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Hold out evenly spaced contiguous slices rather than one tail slice.

    A single trailing slice measures the model on whichever source happens to
    be concatenated last, which is not what validation loss is supposed to
    mean. Spreading the held-out chunks across the stream makes the number
    representative of the corpus as a whole; keeping each chunk contiguous and
    longer than the context window keeps it a fair language-modelling task.
    """
    n_val = max(block_size * 8, int(len(arr) * val_frac))
    chunk = max(block_size + 1, n_val // n_chunks)
    n_chunks = max(1, min(n_chunks, len(arr) // (chunk * 2)))
    stride = len(arr) // n_chunks

    val_parts, train_parts, cursor = [], [], 0
    for i in range(n_chunks):
        # Sit the held-out chunk in the middle of its stride, away from the
        # seams where two sources meet.
        start = i * stride + (stride - chunk) // 2
        end = start + chunk
        train_parts.append(arr[cursor:start])
        val_parts.append(arr[start:end])
        cursor = end
    train_parts.append(arr[cursor:])

    return np.concatenate(train_parts), np.concatenate(val_parts)


# ---------------------------------------------------------------------------
# Chat-format construction
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str, min_words: int = 5, max_words: int = 40) -> Iterator[str]:
    for line in text.split("\n"):
        for s in _SENT_SPLIT.split(line):
            s = s.strip()
            n = s.count(" ") + 1
            if min_words <= n <= max_words and s[:1].isupper():
                yield s


def seed_dialogues(seed_path: Path | None) -> list[list[str]]:
    """Load the hand-written conversation seed shipped with the repo.

    Format: blank-line-separated blocks, alternating `U: ...` / `A: ...` lines.
    """
    if seed_path is None or not seed_path.exists():
        return []
    convos: list[list[str]] = []
    current: list[str] = []
    for line in seed_path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            if len(current) >= 2:
                convos.append(current)
            current = []
            continue
        if line[:2] in ("U:", "A:"):
            current.append(line[2:].strip())
    if len(current) >= 2:
        convos.append(current)
    return convos


def adjacent_pair_dialogues(text: str, limit: int, rng: random.Random) -> list[list[str]]:
    """A weak surrogate for real dialogue data.

    Consecutive sentences become a (prompt, reply) pair. This is not a
    conversation and the model should not be expected to learn conversational
    pragmatics from it — what it *does* teach is the turn protocol and that a
    reply should stay on the topic of the prompt. Genuine conversational skill
    is expected to come from the online learner.
    """
    # Adjacency is the entire signal here, so the sentence order is preserved
    # and `rng` is only threaded through for callers that want reproducibility.
    sents = list(_sentences(text))
    pairs: list[list[str]] = []
    for a, b in zip(sents, sents[1:]):
        pairs.append([a, b])
        if len(pairs) >= limit:
            break
    return pairs


def encode_dialogue(
    tok: BPETokenizer, turns: Sequence[str], block_size: int
) -> tuple[list[int], list[int]] | None:
    """Encode a conversation into (input_ids, targets).

    Targets are IGNORE_INDEX everywhere except on the tokens Aria herself
    produces, so gradient only ever flows from Aria's own words.
    """
    ids: list[int] = [tok.bos_id]
    loss_on: list[bool] = [False]
    for i, turn in enumerate(turns):
        speaker = tok.user_id if i % 2 == 0 else tok.aria_id
        body = tok.encode(" " + turn.strip(), allowed_special=False) + [tok.eot_id]
        ids.append(speaker)
        loss_on.append(False)
        ids.extend(body)
        loss_on.extend([i % 2 == 1] * len(body))

    # Truncate from the *left*. Aria's reply is at the end and is the only part
    # that carries loss, so keeping the head would throw away the entire
    # training signal whenever a user message runs long.
    if len(ids) > block_size + 1:
        ids = ids[-(block_size + 1):]
        loss_on = loss_on[-(block_size + 1):]
        ids[0], loss_on[0] = tok.bos_id, False
    if len(ids) < 8:
        return None

    x = ids[:-1]
    y = [
        ids[i + 1] if loss_on[i + 1] else IGNORE_INDEX
        for i in range(len(ids) - 1)
    ]
    if all(t == IGNORE_INDEX for t in y):
        return None
    return x, y


def build_chat_examples(
    tok: BPETokenizer,
    corpus: str,
    block_size: int,
    seed_path: Path | None,
    n_surrogate: int,
    seed_repeat: int,
    rng: random.Random,
) -> list[tuple[list[int], list[int]]]:
    convos: list[list[str]] = []
    seeds = seed_dialogues(seed_path)
    convos.extend(seeds * max(1, seed_repeat))
    convos.extend(adjacent_pair_dialogues(corpus, n_surrogate, rng))

    examples = []
    for turns in convos:
        enc = encode_dialogue(tok, turns, block_size)
        if enc is not None:
            examples.append(enc)
    rng.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# Preparation entrypoint
# ---------------------------------------------------------------------------


def _seed_path(data_dir: Path) -> Path | None:
    """A seed file inside the chosen data dir wins, so `--data-dir` can override
    it; otherwise use the copy shipped inside the package, which is present
    whether Aria was cloned or pip-installed."""
    local = data_dir / "seed_dialogues.txt"
    if local.exists():
        return local
    packaged = Path(__file__).resolve().parent / "seed_dialogues.txt"
    return packaged if packaged.exists() else None


def _rebuild_chat(data_dir: Path, block_size: int, n_surrogate: int,
                  seed_repeat: int, rng: random.Random, verbose: bool) -> dict:
    tok_path = data_dir / "tokenizer.json"
    corpus_path = data_dir / "corpus.txt"
    if not tok_path.exists() or not corpus_path.exists():
        raise RuntimeError(
            "--chat-only needs a tokenizer.json and corpus.txt already in "
            f"{data_dir}; run a full `prepare` first"
        )
    tok = BPETokenizer.load(tok_path)
    corpus = corpus_path.read_text(encoding="utf-8")
    chat = build_chat_examples(tok, corpus, block_size, _seed_path(data_dir),
                               n_surrogate, seed_repeat, rng)
    with open(data_dir / "chat.pt", "wb") as f:
        pickle.dump(chat, f)

    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"chat_examples": len(chat), "seed_repeat": seed_repeat,
                 "surrogate_dialogues": n_surrogate})
    meta_path.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"rebuilt chat.pt: {len(chat)} examples "
              f"(seed x{seed_repeat}, {n_surrogate} surrogate)", flush=True)
    return meta


def prepare(
    data_dir: str | Path = "data",
    vocab_size: int = 8192,
    block_size: int = 256,
    val_frac: float = 0.005,
    bpe_sample_bytes: int = 6_000_000,
    n_surrogate_dialogues: int = 40_000,
    seed_repeat: int = 100,
    offline: bool = False,
    seed: int = 1337,
    chat_only: bool = False,
    verbose: bool = True,
) -> dict:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if chat_only:
        # Rebuild only chat.pt, reusing the tokenizer and corpus already on
        # disk. Retuning the conversation mix is a minute's work this way
        # instead of re-running BPE and re-encoding five million tokens.
        return _rebuild_chat(data_dir, block_size, n_surrogate_dialogues,
                             seed_repeat, rng, verbose)

    if offline:
        paths = sorted((data_dir / "raw").glob("*.txt"))
        if not paths:
            raise RuntimeError("--offline set but data/raw/ contains no .txt files")
    else:
        paths = download(data_dir)

    if verbose:
        print("building corpus ...", flush=True)
    corpus = build_corpus(paths)
    (data_dir / "corpus.txt").write_text(corpus, encoding="utf-8")

    if verbose:
        print(f"corpus: {len(corpus)/1e6:.1f} MB; training BPE (vocab={vocab_size}) ...",
              flush=True)
    tok = BPETokenizer.train(corpus[:bpe_sample_bytes], vocab_size=vocab_size,
                             verbose=verbose)
    tok.save(data_dir / "tokenizer.json")
    if verbose:
        print(f"  learned {len(tok.merges)} merges -> vocab {tok.vocab_size}", flush=True)

    if verbose:
        print("encoding corpus ...", flush=True)
    # Encode in line blocks to keep peak memory modest.
    ids: list[int] = []
    lines = corpus.split("\n")
    step = 2000
    for i in range(0, len(lines), step):
        ids.extend(tok.encode("\n".join(lines[i : i + step]) + "\n",
                              allowed_special=False))
        if verbose and (i // step) % 25 == 0:
            print(f"  {min(i + step, len(lines))}/{len(lines)} lines", flush=True)

    arr = np.array(ids, dtype=np.uint16)
    train_arr, val_arr = split_train_val(arr, block_size, val_frac)
    train_arr.tofile(data_dir / "train.bin")
    val_arr.tofile(data_dir / "val.bin")

    if verbose:
        print("building chat examples ...", flush=True)
    chat = build_chat_examples(tok, corpus, block_size, _seed_path(data_dir),
                               n_surrogate_dialogues, seed_repeat, rng)
    with open(data_dir / "chat.pt", "wb") as f:
        pickle.dump(chat, f)

    meta = {
        "vocab_size": tok.vocab_size,
        "block_size": block_size,
        "train_tokens": int(train_arr.size),
        "val_tokens": int(val_arr.size),
        "chat_examples": len(chat),
        "seed_repeat": seed_repeat,
        "corpus_chars": len(corpus),
        "sources": [p.name for p in paths],
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(json.dumps(meta, indent=2), flush=True)
    return meta


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TokenStream:
    """Random fixed-length crops from a flat uint16 token file."""

    def __init__(self, path: str | Path, block_size: int) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path} has too few tokens ({len(self.data)})")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, generator: torch.Generator | None = None):
        hi = len(self.data) - self.block_size - 1
        ix = torch.randint(hi, (batch_size,), generator=generator)
        x = torch.stack([
            torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])
        return x, y


class ChatSet:
    """Padded batches of turn-structured examples."""

    def __init__(self, path: str | Path, pad_id: int) -> None:
        with open(path, "rb") as f:
            self.examples: list[tuple[list[int], list[int]]] = pickle.load(f)
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def batch(self, batch_size: int, rng: random.Random):
        picks = [self.examples[rng.randrange(len(self.examples))] for _ in range(batch_size)]
        return collate(picks, self.pad_id)


def collate(examples: Sequence[tuple[Sequence[int], Sequence[int]]], pad_id: int):
    n = max(len(x) for x, _ in examples)
    xs = torch.full((len(examples), n), pad_id, dtype=torch.long)
    ys = torch.full((len(examples), n), IGNORE_INDEX, dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        xs[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        ys[i, : len(y)] = torch.tensor(y, dtype=torch.long)
    return xs, ys


def mixed_batch(
    stream: TokenStream | None,
    chatset: ChatSet | None,
    batch_size: int,
    chat_frac: float,
    rng: random.Random,
    generator: torch.Generator | None,
    pad_id: int,
    block_size: int | None = None,
):
    """Interleave plain-text and chat-format examples in one optimiser step.

    Chat examples were cut to the block size `prepare` was run with, which
    need not be the model's: reusing data prepared for a 256-token model to
    train a 128-token one would otherwise overflow the context. Over-long
    examples keep their tail, where Aria's reply and its loss are."""
    n_chat = int(round(batch_size * chat_frac)) if chatset and len(chatset) else 0
    n_text = batch_size - n_chat
    parts = []
    if n_text > 0 and stream is not None:
        x, y = stream.batch(n_text, generator)
        parts.extend(list(zip(x.tolist(), y.tolist())))
    if n_chat > 0 and chatset is not None:
        picks = [chatset.examples[rng.randrange(len(chatset))] for _ in range(n_chat)]
        for a, b in picks:
            if block_size and len(a) > block_size:
                a, b = a[-block_size:], b[-block_size:]
            parts.append((list(a), list(b)))
    rng.shuffle(parts)
    return collate(parts, pad_id)


def copy_tokenizer(data_dir: Path, out_dir: Path) -> None:
    src = data_dir / "tokenizer.json"
    if src.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out_dir / "tokenizer.json")
