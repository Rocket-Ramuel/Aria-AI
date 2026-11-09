"""Choosing where Aria runs: an NVIDIA GPU, an Apple GPU, or the CPU.

`--device auto` (the default) takes the first that is present and works:

1. ``cuda`` — an NVIDIA GPU with the CUDA build of PyTorch;
2. ``mps``  — the GPU of an Apple Silicon Mac (M1 or newer), through
   PyTorch's Metal backend, for a model big enough to gain from it;
3. ``cpu``  — always available.

A GPU is put through a short self-test before it is trusted: a tiny model
does everything Aria does (a forward and backward pass, an optimiser step,
sampling, memory recall) on it. If any of that fails — an operation the
installed PyTorch doesn't support on that GPU, say — Aria says so and runs on
the CPU instead of crashing halfway through a conversation.

On a Mac, PyTorch is also told to run any operation the Apple GPU lacks on the
CPU (`PYTORCH_ENABLE_MPS_FALLBACK`), so a missing operation is slow rather
than fatal.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

# Must be set before PyTorch's Metal backend starts; harmless elsewhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

CHOICES = ("auto", "cpu", "cuda", "mps")

# Chatting with a small model, an Apple GPU is slower than the CPU: each word
# is a few dozen tiny steps, and handing each one to the GPU costs more than
# the step itself. On GitHub's Apple Silicon machines the shipped 6.5M-parameter
# model took 2.5 s to answer on the GPU against 0.1 s on the same machine's
# CPU, and 10-13 s to read a short document against 3-4 s. So for a model
# smaller than this, `auto` leaves the Apple GPU alone; `--device mps` still
# uses it.
MPS_MIN_PARAMS = 50_000_000
_verified: dict[str, str | None] = {}     # device -> None if fine, else the error


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    # A Mac whose graphics can't run Metal 2 (some virtual machines) has no
    # usable Apple GPU, and PyTorch warns as it finds that out. Answering the
    # question is all that's needed here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(backend.is_available())


def kind(device: str | torch.device) -> str:
    """'cuda:1' -> 'cuda'."""
    return str(device).split(":")[0]


def self_test(device: str) -> str | None:
    """Run a miniature of everything Aria does on `device`.

    Returns None if it all works, or a description of what failed."""
    if device in _verified:
        return _verified[device]
    from .config import ModelConfig
    from .model import GPT
    from .optim import LowMemoryAdam
    from .sample import generate
    try:
        torch.manual_seed(0)
        model = GPT(ModelConfig(vocab_size=64, n_layer=1, n_head=2, n_kv_head=1,
                                n_embd=16, block_size=16, n_areas=2)).to(device)
        model.train()
        model.checkpointing = True
        x = torch.randint(0, 64, (2, 12), device=device)
        for opt in (torch.optim.AdamW(model.parameters(), lr=1e-3),
                    LowMemoryAdam(list(model.parameters()), lr=1e-3)):
            opt.zero_grad()
            loss = model(x, x)[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        model.checkpointing = False
        out = model(x, x, return_hidden=True)
        keys = torch.nn.functional.normalize(out[3][0].float(), dim=-1).half()
        top, idx = (keys.float() @ keys.float()[0]).topk(3)
        values = x[0][idx]                          # what the memory would recall
        torch.zeros(3, 64, device=device).scatter_add_(
            1, values[None].expand(3, -1), torch.softmax(top, 0)[None].expand(3, -1))
        list(generate(model, [1, 2, 3], max_new_tokens=3, device=device))
        result = None
    except Exception as e:                      # anything at all: don't trust it
        result = f"{type(e).__name__}: {e}".splitlines()[0][:200]
    _verified[device] = result
    return result


def resolve(requested: str | None = "auto", verbose: bool = True,
            n_params: int | None = None) -> str:
    """The device to run on, for a `--device` value. `n_params`: the size of
    the model to be chatted with, if known (see MPS_MIN_PARAMS)."""
    requested = (requested or "auto").lower()
    if kind(requested) not in CHOICES:
        raise ValueError(f"unknown device {requested!r}; choose from {', '.join(CHOICES)}")

    if requested == "auto":
        candidates = []
        if torch.cuda.is_available():
            candidates.append("cuda")
        if mps_available() and (n_params is None or n_params >= MPS_MIN_PARAMS):
            candidates.append("mps")
    elif requested == "cpu":
        return "cpu"
    else:
        present = torch.cuda.is_available() if kind(requested) == "cuda" else mps_available()
        if not present:
            why = ("no NVIDIA GPU, or PyTorch was installed without CUDA"
                   if kind(requested) == "cuda" else
                   "this isn't an Apple Silicon Mac, or PyTorch/macOS is too old for Metal")
            _say(verbose, f"--device {requested} isn't available ({why}); using the CPU")
            return "cpu"
        candidates = [requested]

    for device in candidates:
        problem = self_test(device)
        if problem is None:
            return device
        _say(verbose, f"the {_name(device)} failed a self-test ({problem}); using the CPU instead")
    return "cpu"


def describe(device: str) -> str:
    return _name(device)


def _name(device: str) -> str:
    return {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU"}.get(kind(device), device)


def _say(verbose: bool, message: str) -> None:
    if verbose:
        print(f"aria: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# bfloat16 on the CPU
# ---------------------------------------------------------------------------

CHECK_BF16_FLAG = "--check-bfloat16"
_bf16_works: dict[str, bool] = {}


def check_bf16_here() -> None:
    """A miniature of Aria learning in bfloat16 on the CPU. Run in a separate
    process by cpu_bf16_works: if the CPU can't really do it, that process
    dies, not Aria."""
    from .config import ModelConfig
    from .model import GPT
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=64, n_layer=1, n_head=2, n_kv_head=1,
                            n_embd=64, block_size=32, n_areas=2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (4, 32))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(x, x)[1]
        big = torch.nn.Linear(512, 2048)(torch.randn(64, 512))
    (loss + big.float().mean()).backward()
    opt.step()


def _check_command() -> tuple[list[str], str | None]:
    """How to run check_bf16_here in a new process, and in which folder."""
    if getattr(sys, "frozen", False):           # the packaged app
        return [sys.executable, "--aria-check-bfloat16"], None   # packaging/aria_app.py
    return ([sys.executable, "-m", "aria.device", CHECK_BF16_FLAG],
            str(Path(__file__).resolve().parent.parent))


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def cpu_bf16_works() -> bool:
    """Whether bfloat16 arithmetic really runs on this CPU.

    A CPU can say it has bfloat16 instructions and still not run them (seen
    on one of GitHub's Windows machines with PyTorch 2.15). That isn't an
    error Aria could catch: it's an "illegal instruction" that ends the whole
    program. So it's tried once in a separate process, and the answer kept
    for this PyTorch and this processor."""
    from .paths import data_dir
    key = f"{torch.__version__} {platform.machine()} {_cpu_name()}"
    if key in _bf16_works:
        return _bf16_works[key]
    cache = data_dir() / "cpu-check.json"
    try:
        known = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    if key not in known:
        cmd, cwd = _check_command()
        try:
            known[key] = subprocess.run(cmd, cwd=cwd, capture_output=True,
                                        timeout=300).returncode == 0
        except (OSError, subprocess.SubprocessError):
            known[key] = False
        if not known[key]:
            _say(True, "this CPU says it can do bfloat16 arithmetic but can't run "
                       "it here, so learning stays in float32")
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(known, indent=2), encoding="utf-8")
        except OSError:
            pass
    _bf16_works[key] = bool(known[key])
    return _bf16_works[key]


if __name__ == "__main__":
    if sys.argv[1:] == [CHECK_BF16_FLAG]:
        check_bf16_here()
        sys.exit(0)
    sys.exit(f"usage: python -m aria.device {CHECK_BF16_FLAG}")
