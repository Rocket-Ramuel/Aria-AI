"""How Aria's weights are kept on disk, compactly.

A model's file size is fixed by its architecture, not by how much it has
learned: learning changes the values of the weights, never how many there are.
A "super trained" small Aria is exactly as large as a fresh one. What this
module does is make that fixed size as small as it can be without costing
anything the model needs:

* **Half precision.** Weights are stored as float16 and loaded back into a
  float32 model. That halves the file and changes the model's outputs by less
  than the noise of sampling (round-trip error is ~5e-4 relative per weight).
* **No duplicates.** The input embedding and output layer are one tied
  matrix; it is written once, not twice.
* **Fisher information in one byte per weight.** EWC only needs to know
  roughly how important each weight is, so its log is quantised to 8 bits
  (about 8% relative resolution) instead of 16 or 32. Plain float16 is the
  wrong choice here: Fisher values sit around 1e-7, below float16's normal
  range, so float16 rounds many of them to zero.
* **Atomic writes.** A save goes to a temporary file that replaces the old
  one only when complete, so a crash or power cut mid-save can never leave a
  corrupt checkpoint behind.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

FISHER_ENCODING = "log-uint8"
_HALF_MAX = 65504.0


def half_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """float16 copy of a state dict, keeping tied tensors tied.

    Tensors that share memory (tied embeddings) map to one converted tensor,
    so `torch.save` stores it once. A tensor too large for float16's range is
    left in float32 rather than silently overflowing to infinity."""
    out: dict[str, torch.Tensor] = {}
    seen: dict[tuple, torch.Tensor] = {}
    for k, v in sd.items():
        if not v.is_floating_point():
            out[k] = v
            continue
        key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape),
               tuple(v.stride()))
        if key not in seen:
            fits = v.numel() == 0 or float(v.detach().abs().max()) < _HALF_MAX
            seen[key] = v.detach().half() if fits else v.detach().float()
        out[k] = seen[key]
    return out


def encode_fisher(fisher: dict[str, torch.Tensor] | None) -> dict[str, Any] | None:
    """Quantise Fisher information to one byte per weight, on a log scale.

    Code 0 is exactly zero; codes 1..255 span [min, max] of the positive
    values logarithmically."""
    if not fisher:
        return None
    lo, hi = float("inf"), float("-inf")
    for v in fisher.values():
        pos = v[v > 0].float()
        if pos.numel():
            lo = min(lo, float(pos.log().min()))
            hi = max(hi, float(pos.log().max()))
    if lo == float("inf"):
        lo = hi = 0.0
    span = max(hi - lo, 1e-12)
    q = {}
    for k, v in fisher.items():
        v = v.float()
        codes = torch.zeros(v.shape, dtype=torch.uint8)
        pos = v > 0
        codes[pos] = (((v[pos].log() - lo) / span) * 254).round().clamp(0, 254).to(torch.uint8) + 1
        q[k] = codes
    return {"encoding": FISHER_ENCODING, "lo": lo, "hi": hi, "q": q}


def decode_fisher(obj: Any) -> dict[str, torch.Tensor] | None:
    """Fisher as float32 tensors, whichever way it was stored."""
    if not obj:
        return None
    if isinstance(obj, dict) and obj.get("encoding") == FISHER_ENCODING:
        lo, hi = float(obj["lo"]), float(obj["hi"])
        span = max(hi - lo, 1e-12)
        out = {}
        for k, codes in obj["q"].items():
            vals = torch.exp(lo + (codes.float() - 1) / 254 * span)
            out[k] = torch.where(codes == 0, torch.zeros_like(vals), vals)
        return out
    return {k: v.float() for k, v in obj.items()}


def atomic_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def dir_size_mb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def process_ram_mb() -> float | None:
    """Peak resident memory of this process, where the OS reports it."""
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS bytes.
        return peak / (1e6 if sys.platform == "darwin" else 1e3)
    except (ImportError, OSError):
        return None
