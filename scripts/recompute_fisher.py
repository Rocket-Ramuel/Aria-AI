"""Recompute a checkpoint's Fisher information and re-save it compactly.

The shipped checkpoint was exported with its Fisher information in float16,
which cannot represent values around 1e-7: 11% of them rounded to zero and
most of the rest kept only a few bits. It was also estimated from squared
*batch* gradients rather than per-sequence ones. This script re-estimates it
the way `aria.pretrain.estimate_fisher` now does, from the same public corpus,
using the checkpoint's own tokenizer, and writes the checkpoint back in the
compact format (`aria.storage`): float16 weights with the tied embedding
stored once, Fisher as one byte per weight. The model's weights are unchanged.

    python scripts/recompute_fisher.py checkpoints/aria-small.pt

Needs network access to fetch the corpus (or a prepared `--data-dir`).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.data import TokenStream, build_corpus, download
from aria.pretrain import estimate_fisher, load_checkpoint
from aria.storage import atomic_save, encode_fisher, half_state_dict


def sample_corpus(text: str, n_pieces: int = 48, piece_chars: int = 64_000) -> str:
    """Evenly spaced slices, so the estimate covers every source."""
    step = max(1, len(text) // n_pieces)
    return "\n".join(text[i : i + piece_chars] for i in range(0, len(text), step))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data-dir", default=None, help="reuse an existing data/raw")
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    path = Path(args.checkpoint)
    model, tok, cfg, ckpt = load_checkpoint(path)
    before = path.stat().st_size

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(args.data_dir or tmp)
        raw = sorted((data_dir / "raw").glob("*.txt")) or download(data_dir)
        corpus = sample_corpus(build_corpus(raw))
        print(f"encoding {len(corpus) / 1e6:.1f} MB of corpus with the checkpoint's tokenizer")
        ids = np.array(tok.encode(corpus, allowed_special=False), dtype=np.uint16)
        ids.tofile(Path(tmp) / "sample.bin")
        stream = TokenStream(Path(tmp) / "sample.bin", cfg.model.block_size)
        print(f"estimating Fisher over {args.batches * args.batch_size} sequences")
        fisher = estimate_fisher(model, stream, args.batches, args.batch_size,
                                 torch.Generator().manual_seed(1337))

    zeros = sum(int((v == 0).sum()) for v in fisher.values())
    total = sum(v.numel() for v in fisher.values())
    print(f"zero entries: {100 * zeros / total:.2f}%")
    atomic_save({
        "model": half_state_dict(model.state_dict()),
        "config": ckpt["config"],
        "tokenizer": ckpt["tokenizer"],
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "fisher": encode_fisher(fisher),
    }, path)
    print(f"{path}: {before / 1e6:.1f} MB -> {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
