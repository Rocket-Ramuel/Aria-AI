"""Autoregressive sampling with a KV cache."""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import torch
import torch.nn.functional as F

from .model import GPT
from .tokenizer import BPETokenizer


def _filter_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    if top_k:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        drop = probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model: GPT,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 96,
    temperature: float = 0.85,
    top_k: int | None = 40,
    top_p: float | None = 0.92,
    repetition_penalty: float = 1.12,
    no_repeat_window: int = 64,
    stop_ids: Sequence[int] = (),
    device: torch.device | str = "cpu",
    memory=None,
) -> Iterator[int]:
    """Yield generated token ids one at a time.

    The prompt is truncated from the left to fit the context window, so a long
    conversation degrades into a sliding window rather than raising.
    """
    model.eval()
    block = model.cfg.block_size
    ids = list(prompt_ids)[-(block - 1):]
    if not ids:
        raise ValueError("prompt is empty")

    # `memory` is a Hippocampus: it recalls the episodes this conversation is
    # about, then blends what came next in them into each prediction.
    if memory is not None:
        memory.focus(ids[-64:])

    def forward(inp, caches):
        if memory is None:
            logits, _, caches = model(inp, kv_caches=caches)
            return logits[:, -1, :].float(), caches
        logits, _, caches, hidden = model(inp, kv_caches=caches, return_hidden=True)
        return memory.recall(hidden[0, -1:], logits[0, -1:]), caches

    x = torch.tensor([ids], dtype=torch.long, device=device)
    caches = model.empty_cache()
    step, caches = forward(x, caches)

    generated: list[int] = []
    stop = set(stop_ids)

    for _ in range(max_new_tokens):
        step_logits = step.clone()   # float32, even under bfloat16 autocast

        if repetition_penalty != 1.0:
            recent = (ids + generated)[-no_repeat_window:]
            if recent:
                idx = torch.tensor(sorted(set(recent)), device=device)
                vals = step_logits[0, idx]
                step_logits[0, idx] = torch.where(
                    vals > 0, vals / repetition_penalty, vals * repetition_penalty
                )

        step_logits = _filter_logits(step_logits, temperature, top_k, top_p)
        probs = F.softmax(step_logits, dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1).item())

        if nxt in stop:
            return
        yield nxt
        generated.append(nxt)

        # Context is full: drop the cache and re-prime on a right-aligned window.
        if caches[0][0].shape[2] + 1 >= block:
            ids = (ids + generated)[-(block // 2):]
            generated = []
            caches = model.empty_cache()
            x = torch.tensor([ids], dtype=torch.long, device=device)
            step, caches = forward(x, caches)
            continue

        x = torch.tensor([[nxt]], dtype=torch.long, device=device)
        step, caches = forward(x, caches)


def complete(
    model: GPT,
    tok: BPETokenizer,
    prompt: str,
    device: torch.device | str = "cpu",
    **kwargs,
) -> str:
    ids = tok.encode(prompt)
    out = list(generate(model, ids, device=device, **kwargs))
    return tok.decode(out, skip_special=True)


def build_chat_prompt(
    tok: BPETokenizer,
    history: Sequence[tuple[str, str]],
    user_message: str,
    block_size: int,
) -> list[int]:
    """Render conversation history into the training-time chat format.

    History is dropped from the oldest end until the prompt fits, leaving room
    for the reply."""
    reserve = min(160, block_size // 3)
    budget = block_size - reserve

    tail: list[int] = [tok.user_id]
    tail += tok.encode(" " + user_message.strip(), allowed_special=False)
    tail += [tok.eot_id, tok.aria_id]

    turns: list[list[int]] = []
    for who, text in history:
        marker = tok.user_id if who == "user" else tok.aria_id
        turns.append([marker] + tok.encode(" " + text.strip(), allowed_special=False)
                     + [tok.eot_id])

    prompt = [tok.bos_id] + tail
    kept: list[list[int]] = []
    used = len(prompt)
    for turn in reversed(turns):
        if used + len(turn) > budget:
            break
        kept.insert(0, turn)
        used += len(turn)

    return [tok.bos_id] + [t for turn in kept for t in turn] + tail
