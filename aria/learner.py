"""The online learner: Aria's weights change while you talk to her.

Doing this naively — one SGD step per turn on whatever the user just typed — is
the fastest known way to destroy a language model. Within a few dozen turns it
will parrot its last input, and within a few hundred it will have forgotten how
to form a sentence. Every mechanism in this file exists to make continuous
learning survivable:

1. **Surprise gating.** Compute the loss on the new exchange first. If the model
   already predicts it well, there is nothing to learn — skip the update. Only
   novel material produces a gradient step, and the step size scales with how
   novel it was.

2. **Rehearsal.** Each update batch mixes the new exchange with samples drawn
   from a persistent replay buffer of past conversations *and* from the original
   pretraining corpus. The model never sees a batch of only-new-data, which is
   the condition under which catastrophic forgetting happens.

3. **Elastic weight consolidation.** The diagonal Fisher information estimated
   at the end of pretraining says which weights the model's existing knowledge
   is sensitive to. Those weights are pulled back toward their anchor values in
   proportion to that sensitivity; insensitive weights move freely.

4. **A trust region.** After every step each tensor is projected back so it can
   never drift more than `trust_radius` (relative) from its anchor. The anchor
   moves at each consolidation, so this bounds the drift *between*
   consolidations; across many of them the canary is what bounds it.

5. **A canary and rollback.** A fixed set of held-out English sentences is
   evaluated periodically. If loss on them rises past tolerance, the learner
   restores the last known-good snapshot and lowers its learning rate. Learning
   that makes the model worse is undone automatically.

6. **Consolidation ("sleep").** Periodically the current weights are accepted as
   the new known-good state and the anchor is moved toward them. In LoRA mode
   the adapters are folded into the base weight matrices at this point and reset
   to zero — which is what makes the learning permanent rather than a growing
   pile of adapters.

Two things are learned from each exchange: Aria's reply, and — when
`style_mirror` is on — the user's own message, framed as if Aria had said it.
The second is what makes her drift toward the voice of the person she talks
to. Uploaded documents and transcripts go through `iter_learn_units` and
`iter_learn_dialogues`, which stream the material from disk and take many steps
over it instead of one. There is no limit on how much: a document is never
held in memory whole, and learning runs until every pass is done.

A model created blank (`aria.pretrain.create_blank_checkpoint`) runs with
`blank_learner_config()`, which switches off the safeguards that only exist to
protect pretrained knowledge.
"""

from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Iterator, Optional, Sequence

import torch

from .config import LearnerConfig
from .data import TokenStream, collate, encode_dialogue, encode_dialogue_ids
from .documents import iter_units
from .hippocampus import Hippocampus
from .memory import Journal, ReplayBuffer, canary_texts
from .model import (GPT, IGNORE_INDEX, LoRALinear, attach_lora, block_index,
                    lora_parameters, merge_lora, merged_state_dict)
from .optim import LowMemoryAdam
from .storage import atomic_save, decode_fisher, dir_size_mb, half_state_dict
from .tokenizer import BPETokenizer

FFN_KEYS = ("gate_proj", "up_proj", "down_proj")

# Above this many parameters, "auto" memory saving switches on: the low-memory
# optimiser, activation checkpointing, and half-precision safety snapshots.
MEMORY_SAVER_PARAMS = 20_000_000

Example = tuple[list[int], list[int]]

# How many chunks of one uploaded document are kept for later rehearsal. Enough
# to keep the lesson alive, few enough that one book can't evict every
# conversation from the reservoir.
DOCUMENT_REPLAY_CHUNKS = 48

# Held-out text: every HOLDOUT_EVERY-th run of HOLDOUT_SEGMENT sentences is
# never trained on. Loss on it says whether Aria learned the *language* of a
# document — grammar, word order, vocabulary — rather than memorising it.
HOLDOUT_EVERY = 20
HOLDOUT_SEGMENT = 8
PROBE_SIZE = 32
# Examples are shuffled within a buffer of this many as they stream past, so
# consecutive steps don't all come from the same page.
SHUFFLE_BUFFER = 256
# The canary may roll back a step that hurt general English; one upload is
# allowed this many before the learner decides the material itself is the
# problem and stops.
MAX_ROLLBACKS_PER_UPLOAD = 3


@dataclass
class LearnProgress:
    """Where a document upload has got to; yielded after every step."""

    phase: str            # "reading" (first pass, counting) or "learning"
    step: int = 0
    total: int = 0
    examples: int = 0
    loss: Optional[float] = None


Progress = Optional[Callable[[LearnProgress], None]]
StopCheck = Optional[Callable[[], bool]]


class _Reservoir:
    """A uniform sample of at most `k` items from a stream of unknown length."""

    def __init__(self, k: int, rng: random.Random) -> None:
        self.k, self.rng, self.seen, self.items = k, rng, 0, []

    def add(self, item) -> None:
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen)
            if j < self.k:
                self.items[j] = item


class _UnitEncoder:
    """Turns a stream of sentences into training examples, incrementally.

    Two kinds come out:

    * **windows** — the running text cut into context-sized, overlapping
      pieces, learned as continuous prose in Aria's voice;
    * **exchanges** — each sentence as the reply to the one before it, which
      puts the same voice where replies are generated (after
      `<user> ... <eot><aria>`). For a model with no pretraining this is the
      difference between memorising a sample and answering in its style.
    """

    def __init__(self, learner: "OnlineLearner") -> None:
        self.tok = learner.tok
        self.block = learner.model.cfg.block_size
        self.span = self.block - 2                   # room for <bos><aria>
        self.stride = max(1, (self.span * 3) // 4)
        self.buf: list[int] = []
        self.fresh = 0                               # tokens in no window yet
        self.prev: list[int] | None = None

    def _window(self, chunk: list[int], end: bool) -> Example:
        seq = [self.tok.bos_id, self.tok.aria_id] + chunk
        if end:
            seq.append(self.tok.eot_id)
        x, y = seq[:-1], seq[1:]
        y[0] = IGNORE_INDEX
        return x, y

    def feed(self, ids: list[int]) -> Iterator[tuple[str, Example]]:
        if self.prev is not None:
            half = self.block // 2
            ex = encode_dialogue_ids(self.tok, [self.prev[-half:], ids[:half]], self.block)
            if ex is not None:
                yield "exchange", ex
        self.prev = ids
        self.buf.extend(ids)
        self.fresh += len(ids)
        while len(self.buf) >= self.span:
            yield "window", self._window(self.buf[: self.span], end=False)
            del self.buf[: self.stride]
            self.fresh = max(0, len(self.buf) - (self.span - self.stride))

    def flush(self) -> Iterator[tuple[str, Example]]:
        if self.fresh > 0 and len(self.buf) >= 6:
            yield "window", self._window(self.buf, end=True)
        self.buf, self.fresh = [], 0


@dataclass
class UpdateReport:
    """What the learner did with one exchange (or one uploaded document)."""

    applied: bool
    reason: str
    loss_before: float
    loss_after: float
    surprise: float
    lr: float
    grad_norm: float = 0.0
    # None when nothing anchors the weights (a blank model), so there is no
    # distance to report.
    drift: Optional[float] = 0.0
    replayed: int = 0
    rolled_back: bool = False
    consolidated: bool = False
    canary: Optional[float] = None
    steps: int = 1
    # Uploads only:
    heldout: bool = False     # loss measured on text that was never trained on
    stopped: bool = False     # ended early on request
    examples: int = 0
    words: int = 0

    def line(self) -> str:
        if not self.applied:
            return f"[learn] skipped ({self.reason}); loss {self.loss_before:.3f}"
        label = "held-out loss" if self.heldout else "loss"
        bits = [
            f"[learn] {label} {self.loss_before:.3f} -> {self.loss_after:.3f}",
            f"surprise {self.surprise:+.2f}",
            f"lr {self.lr:.1e}",
            f"replay {self.replayed}",
        ]
        if self.drift is not None:
            bits.append(f"drift {self.drift:.3f}")
        if self.steps > 1:
            bits.insert(1, f"{self.steps} steps")
        if self.canary is not None:
            bits.append(f"canary {self.canary:.3f}")
        if self.consolidated:
            bits.append("consolidated")
        if self.rolled_back:
            bits.append("ROLLED BACK")
        if self.stopped:
            bits.append("stopped early")
        return "  ".join(bits)


class OnlineLearner:
    def __init__(
        self,
        model: GPT,
        tok: BPETokenizer,
        cfg: LearnerConfig,
        fisher: dict[str, torch.Tensor] | None = None,
        pretrain_stream: TokenStream | None = None,
        state_dir: str | Path = "runs/aria/online",
        device: str = "cpu",
        extra_canaries: Sequence[str] = (),
        grown_from: int | None = None,
        prior_tokens: int = 0,
    ) -> None:
        self.model = model
        # Blocks at or above this index were added by growth. They start as
        # exact no-ops, so they have nothing to protect: they are always fully
        # trainable, outside LoRA, the anchors and the trust region.
        self.grown_from = grown_from if grown_from is not None and \
            grown_from < model.cfg.n_layer else None
        mode = cfg.memory_saver
        if mode not in ("auto", "on", "off"):
            raise ValueError(f"memory_saver must be auto, on or off, not {mode!r}")
        self.saving_memory = mode == "on" or (
            mode == "auto" and model.num_params() > MEMORY_SAVER_PARAMS)
        model.checkpointing = self.saving_memory
        # bfloat16 arithmetic (weights stay float32): 2.4x faster learning on a
        # CPU with native bfloat16 (measured, AMX), but slower on one without,
        # so only where the hardware has it.
        self.fast_math = self.saving_memory and bf16_supported(device)
        self.tok = tok
        self.cfg = cfg
        self.device = device
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(cfg.seed)
        self.torch_gen = torch.Generator().manual_seed(cfg.seed)
        self.pretrain_stream = pretrain_stream

        self.replay = ReplayBuffer.load(
            self.state_dir / "replay.json", capacity=cfg.replay_capacity, seed=cfg.seed
        )
        self.journal = Journal(self.state_dir / "journal.jsonl")

        self._configure_plasticity()
        self.fisher = self._restrict_fisher(decode_fisher(fisher))
        # Each of these is a full copy of the trainable weights, so they are
        # only kept when something uses them: the anchor for the trust region,
        # L2 pull and EWC (none of which a blank model has), the last good
        # state for rollback (which needs the canary). For a blank model that
        # is two fewer copies of the whole network in memory.
        self.anchor = self._snapshot() if self._uses_anchor() else {}
        self.last_good = self._snapshot() if cfg.health_check else {}

        self.opt = self._new_optimizer(cfg.learning_rate)

        self.canaries = self._encode_canaries(canary_texts(extra_canaries))
        self.canary_baseline = self.canary_loss()
        self.lr = cfg.learning_rate
        self.ema_loss: float | None = None
        self.updates_applied = 0
        self.turns_seen = 0
        self.consecutive_bad = 0
        # Counters rather than `updates_applied % interval`, so that a document
        # upload taking many steps can't jump over a scheduled consolidation.
        self.since_health = 0
        self.since_consolidation = 0
        # Tokens the model has been trained on since it was created; how full
        # it is getting, for growth (see `room_left`).
        self.tokens_learned = 0
        self.prior_tokens = prior_tokens      # e.g. what pretraining used

        self._load_runtime_state()

        # The fast memory system. Built from the replay buffer's text, encoded
        # by the cortex as it is now (see aria.hippocampus).
        self.hippocampus = None
        if cfg.hippocampus_tokens > 0:
            self.hippocampus = Hippocampus(
                model, tok, capacity_tokens=cfg.hippocampus_tokens,
                threshold=cfg.recall_threshold, strength=cfg.recall_strength,
                device=device)
            self.hippocampus.rebuild(self.replay.items)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def fast(self):
        """Context for the forward pass of learning and generation: bfloat16
        arithmetic when `fast_math` is on, otherwise nothing. Evaluations that
        the safety net compares (canary, losses in reports) stay float32."""
        if not self.fast_math:
            return nullcontext()
        kind = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return torch.autocast(device_type=kind, dtype=torch.bfloat16)

    def _is_grown(self, name: str) -> bool:
        idx = block_index(name)
        return self.grown_from is not None and idx is not None and idx >= self.grown_from

    def _configure_plasticity(self) -> None:
        mode = self.cfg.plasticity
        if mode == "lora":
            n = attach_lora(self.model, self.cfg.lora_rank, self.cfg.lora_alpha,
                            below_layer=self.grown_from)
            if n == 0:
                raise RuntimeError("no LoRA targets found in model")
            for p in self.model.parameters():
                p.requires_grad_(False)
            for p in lora_parameters(self.model):
                p.requires_grad_(True)
        elif mode == "ffn":
            for name, p in self.model.named_parameters():
                p.requires_grad_(any(k in name for k in FFN_KEYS))
        elif mode == "full":
            for p in self.model.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"unknown plasticity mode {mode!r}")
        for name, p in self.model.named_parameters():
            if self._is_grown(name):
                p.requires_grad_(True)
        # New adapters are created on the CPU; every mode re-asserts the device
        # so a consolidation never leaves part of the model behind.
        self.model.to(self.device)

        self.trainable_names = [n for n, p in self.model.named_parameters()
                                if p.requires_grad]
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not self.trainable:
            raise RuntimeError(f"plasticity={mode!r} left no trainable parameters")

    def _new_optimizer(self, lr: float):
        if self.saving_memory:
            return LowMemoryAdam(self.trainable, lr=lr,
                                 betas=(self.cfg.beta1, self.cfg.beta2))
        return torch.optim.AdamW(
            self.trainable, lr=lr,
            betas=(self.cfg.beta1, self.cfg.beta2), weight_decay=0.0,
        )

    def _restrict_fisher(self, fisher):
        if not fisher or self.cfg.plasticity == "lora":
            # LoRA adapters have no pretrained counterpart, so there is no
            # Fisher for them; `l2_anchor` (a pull toward zero) does that job.
            return {}
        named = dict(self.model.named_parameters())
        out = {}
        for n in self.trainable_names:
            f = fisher.get(n)
            if f is not None and f.shape == named[n].shape:
                # .float() because a half-precision export would otherwise
                # silently make the EWC penalty half precision too.
                out[n] = f.to(device=self.device, dtype=torch.float32)
        if out:
            # Normalise so ewc_lambda means the same thing across runs.
            total = sum(float(v.mean()) for v in out.values()) / len(out)
            if total > 0:
                out = {k: v / total for k, v in out.items()}
        return out

    def _uses_anchor(self) -> bool:
        if self.cfg.plasticity == "lora":
            return False      # the anchor of an adapter is zero
        return (self.cfg.trust_radius > 0 or self.cfg.l2_anchor > 0
                or (self.cfg.ewc_lambda > 0 and bool(self.fisher)))

    def _snapshot(self) -> dict[str, torch.Tensor]:
        named = dict(self.model.named_parameters())
        # In memory-saving mode the copies are half precision: a rollback
        # then restores weights to within ~5e-4 relative, far inside what the
        # canary can tell apart, for half the memory.
        dtype = torch.float16 if self.saving_memory else None
        return {n: named[n].detach().to(dtype=dtype, copy=True)
                for n in self.trainable_names if not self._is_grown(n)}

    def _restore(self, snap: dict[str, torch.Tensor]) -> None:
        named = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, v in snap.items():
                if n in named:
                    named[n].copy_(v)

    def _encode_canaries(self, texts: Sequence[str]):
        examples = []
        for t in texts:
            ids = [self.tok.bos_id] + self.tok.encode(t, allowed_special=False)
            if len(ids) < 4:
                continue
            examples.append((ids[:-1], ids[1:]))
        return collate(examples, self.tok.pad_id)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def canary_loss(self) -> float:
        was_training = self.model.training
        self.model.eval()
        x, y = self.canaries
        x, y = x[:, : self.model.cfg.block_size], y[:, : self.model.cfg.block_size]
        _, loss, _ = self.model(x.to(self.device), y.to(self.device))
        if was_training:
            self.model.train()
        return float(loss)

    @torch.no_grad()
    def sequence_loss(self, example: Example) -> float:
        return self.mean_loss([example])

    @torch.no_grad()
    def mean_loss(self, examples: Sequence[Example], chunk: int = 4) -> float:
        """Token-averaged loss over some examples.

        Evaluated a few at a time: the logits of one batch are batch x length
        x vocabulary floats, so 32 full-length examples at once would briefly
        need over half a gigabyte for a pretrained model's vocabulary."""
        was_training = self.model.training
        self.model.eval()
        total, tokens = 0.0, 0
        for i in range(0, len(examples), chunk):
            x, y = self._batch(examples[i : i + chunk])
            _, loss, _ = self.model(x, y, loss_reduction="sum")
            total += float(loss)
            tokens += int((y != IGNORE_INDEX).sum())
        if was_training:
            self.model.train()
        return total / max(1, tokens)

    # ------------------------------------------------------------------
    # encoding
    # ------------------------------------------------------------------

    def _encode_turns(self, turns: Sequence[str]) -> Example | None:
        return encode_dialogue(self.tok, turns, self.model.cfg.block_size)

    def _text_example(self, text: str, min_tokens: int = 8) -> Example | None:
        """`text` framed as something Aria said: <bos><aria> text <eot>.

        This is how a document, or the user's own message, shapes Aria's
        voice. Predicting the <aria> marker itself is not part of the lesson.
        """
        body = self.tok.encode(" " + text.strip(), allowed_special=False)
        ids = [self.tok.bos_id, self.tok.aria_id] + body + [self.tok.eot_id]
        ids = ids[: self.model.cfg.block_size + 1]
        if len(ids) < min_tokens:
            return None
        x, y = ids[:-1], ids[1:]
        y[0] = IGNORE_INDEX
        return x, y

    def _style_example(self, turns: Sequence[str]) -> Example | None:
        """The user's latest message, as a lesson in how Aria should sound."""
        if len(turns) < 2:
            return None
        return self._text_example(turns[-2], min_tokens=5)

    def _units_examples(self, units: Iterable[str]) -> Iterator[tuple[str, Example, bool]]:
        """(kind, example, held_out) for a stream of sentences.

        Held-out sentences go through their own encoder, so no training
        window or exchange ever contains a word of them."""
        train, held = _UnitEncoder(self), _UnitEncoder(self)
        for i, unit in enumerate(units):
            ids = self.tok.encode(" " + unit, allowed_special=False)
            is_held = (i // HOLDOUT_SEGMENT) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1
            for kind, ex in (held if is_held else train).feed(ids):
                yield kind, ex, is_held
        for kind, ex in train.flush():
            yield kind, ex, False
        for kind, ex in held.flush():
            yield kind, ex, True

    def _document_windows(self, text: str) -> list[Example]:
        """Every window of a (short) text, held-out logic aside. For tests
        and quick probes; uploads stream through `iter_learn_units`."""
        enc = _UnitEncoder(self)
        out = []
        for unit in iter_units(text.splitlines()):
            out += [ex for k, ex in enc.feed(self.tok.encode(" " + unit, allowed_special=False))
                    if k == "window"]
        return out + [ex for _, ex in enc.flush()]

    def _document_exchanges(self, text: str) -> list[Example]:
        enc = _UnitEncoder(self)
        return [ex for unit in iter_units(text.splitlines())
                for k, ex in enc.feed(self.tok.encode(" " + unit, allowed_special=False))
                if k == "exchange"]

    def _replay_item_example(self, item: dict) -> Example | None:
        turns = item["turns"]
        kind = item.get("kind", "chat")
        if kind == "document":
            return self._text_example(turns[-1])
        if kind == "chat":
            # The stored reply is Aria's own sample. When she isn't meant to
            # learn from those, rehearse the user's words instead.
            if not self.cfg.learn_own_replies:
                return self._style_example(turns)
            if self.cfg.style_mirror and self.rng.random() < 0.5 * self.cfg.style_weight:
                style = self._style_example(turns)
                if style is not None:
                    return style
        return self._encode_turns(turns)

    def _replay_examples(self, n: int) -> list[Example]:
        """Half from past conversations, half from the pretraining corpus.

        The corpus half is what actually anchors English; the conversation half
        is what stops older lessons from being overwritten by newer ones."""
        if n <= 0:
            return []
        n_pre = int(round(n * self.cfg.pretrain_replay_frac)) if self.pretrain_stream else 0
        n_chat = n - n_pre

        out: list[Example] = []
        for item in self.replay.sample(n_chat):
            enc = self._replay_item_example(item)
            if enc is not None:
                out.append(enc)
            if len(out) >= n_chat:
                break

        if n_pre > 0 and self.pretrain_stream is not None:
            xb, yb = self.pretrain_stream.batch(n_pre, self.torch_gen)
            out.extend([(x.tolist(), y.tolist()) for x, y in zip(xb, yb)])
        return out

    def _batch(self, examples: Sequence[Example]):
        x, y = collate(examples, self.tok.pad_id)
        block = self.model.cfg.block_size
        return x[:, :block].to(self.device), y[:, :block].to(self.device)

    # ------------------------------------------------------------------
    # regularisation
    # ------------------------------------------------------------------

    def _anchor_penalty(self) -> torch.Tensor:
        named = dict(self.model.named_parameters())
        total = torch.zeros((), device=self.device)
        if self.cfg.l2_anchor == 0 and (self.cfg.ewc_lambda == 0 or not self.fisher):
            return total
        for n in self.trainable_names:
            p = named[n]
            if self.cfg.plasticity == "lora":
                if "lora_" not in n:
                    continue      # a grown block: nothing to anchor to
                # Anchor is the zero adapter: keep the correction small.
                total = total + self.cfg.l2_anchor * p.pow(2).sum()
                continue
            if n not in self.anchor:
                continue
            delta = p - self.anchor[n].to(p.dtype)
            total = total + self.cfg.l2_anchor * delta.pow(2).sum()
            f = self.fisher.get(n)
            if f is not None:
                total = total + self.cfg.ewc_lambda * (f * delta.pow(2)).sum()
        return total

    @torch.no_grad()
    def _project_trust_region(self) -> float | None:
        """Clip each tensor back inside its allowed distance from the anchor.

        Returns the largest relative drift observed (after projection). A
        `trust_radius` of 0 disables the projection but still reports drift,
        when there is an anchor to measure it from."""
        radius = self.cfg.trust_radius
        clip = radius > 0
        max_drift = 0.0
        named = dict(self.model.named_parameters())

        if self.cfg.plasticity == "lora":
            for module in self.model.modules():
                if not isinstance(module, LoRALinear):
                    continue
                delta = (module.lora_B @ module.lora_A) * module.scaling
                rel = float(delta.norm()) / (float(module.base.weight.norm()) + 1e-12)
                if clip and rel > radius:
                    module.lora_B.mul_(radius / rel)
                    rel = radius
                max_drift = max(max_drift, rel)
            return max_drift

        if not self.anchor:
            return None
        for n in self.trainable_names:
            if n not in self.anchor:
                continue
            p = named[n]
            a = self.anchor[n].to(p.dtype)
            delta = p - a
            rel = float(delta.norm()) / (float(a.norm()) + 1e-12)
            if clip and rel > radius:
                p.copy_(a + delta * (radius / rel))
                rel = radius
            max_drift = max(max_drift, rel)
        return max_drift

    # ------------------------------------------------------------------
    # the update
    # ------------------------------------------------------------------

    def _step(self, batch: Sequence[Example], lr: float) -> tuple[float, float | None]:
        """One optimiser step on `batch`, then the trust-region projection."""
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.model.train()
        x, y = self._batch(batch)
        self.opt.zero_grad(set_to_none=True)
        with self.fast():
            _, ce, _ = self.model(x, y)
        (ce + self._anchor_penalty()).backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.trainable, self.cfg.grad_clip))
        self.opt.step()
        # Free the gradients now rather than at the next step: between
        # updates they would sit in memory as one more copy of the network.
        self.opt.zero_grad(set_to_none=True)
        self.tokens_learned += int((y != IGNORE_INDEX).sum())
        return grad_norm, self._project_trust_region()

    def _after_update(self, report: UpdateReport) -> None:
        """Health check and consolidation, on their own schedules."""
        self.updates_applied += 1
        self.since_health += 1
        self.since_consolidation += 1
        if self.cfg.health_check and self.since_health >= self.cfg.health_interval:
            self.since_health = 0
            report.canary = self._health_check(report)
        if (self.since_consolidation >= self.cfg.consolidate_interval
                and not report.rolled_back):
            self.since_consolidation = 0
            report.consolidated = self.consolidate()

    def observe(
        self,
        turns: Sequence[str],
        weight: float = 1.0,
        kind: str = "chat",
        force: bool = False,
    ) -> UpdateReport:
        """Learn from one exchange.

        `turns` alternates user, aria, user, aria, ...; loss is taken only on
        Aria's turns. `weight` marks how much this exchange matters (corrections
        are worth more than greetings). `force` bypasses the surprise gate.
        """
        self.turns_seen += 1
        reply = self._encode_turns(turns)
        # A correction is the user's own wording, so it is always learned; an
        # ordinary reply only when the config says Aria's own words count.
        if not (self.cfg.learn_own_replies or kind != "chat"):
            reply = None
        style = self._style_example(turns) if self.cfg.style_mirror else None
        lesson = [e for e in (reply, style) if e is not None]
        if not lesson:
            return UpdateReport(False, "nothing to learn from", 0.0, 0.0, 0.0, self.lr)

        loss_before = self.mean_loss(lesson)
        if self.ema_loss is None:
            self.ema_loss = loss_before
        surprise = loss_before - self.ema_loss
        rel_surprise = surprise / max(self.ema_loss, 1e-6)

        # Remember it regardless of whether we take a step: something
        # unremarkable now may still be worth rehearsing later.
        self.replay.add(turns, weight=weight, kind=kind)
        # An episode is the moment itself — the latest exchange — not the
        # whole context window it arrived in, which would put every recent
        # fact into every memory.
        self._remember(list(turns)[-2:], kind)

        gated_out = (
            self.cfg.surprise_gate
            and not force
            and loss_before < self.ema_loss * (1.0 + self.cfg.surprise_floor)
        )
        self.ema_loss = (self.cfg.surprise_ema * self.ema_loss
                         + (1 - self.cfg.surprise_ema) * loss_before)

        if gated_out:
            report = UpdateReport(False, "already predicted well", loss_before,
                                  loss_before, surprise, self.lr)
            self.journal.write(event="update", applied=False, reason=report.reason,
                               loss_before=loss_before, surprise=surprise,
                               turns=list(turns))
            return report

        lr = self.lr * min(
            self.cfg.max_lr_scale,
            max(1.0, 1.0 + self.cfg.lr_surprise_scale * rel_surprise),
        ) * max(0.1, weight)

        replay_examples = self._replay_examples(self.cfg.replay_batch)
        # Next to a reply, the style lesson joins the batch with probability
        # `style_weight` rather than through loss scaling, which keeps the loss
        # a plain token average like everywhere else.
        batch = list(lesson) + replay_examples
        if style is not None and reply is not None and self.cfg.style_weight < 1.0:
            if self.rng.random() > self.cfg.style_weight:
                batch.remove(style)

        grad_norm = drift = 0.0
        for _ in range(max(1, self.cfg.steps_per_turn)):
            grad_norm, drift = self._step(batch, lr)
        loss_after = self.mean_loss(lesson)

        report = UpdateReport(
            applied=True, reason="learned", loss_before=loss_before,
            loss_after=loss_after, surprise=surprise, lr=lr,
            grad_norm=grad_norm, drift=drift, replayed=len(replay_examples),
        )
        self._after_update(report)
        self.journal.write(
            event="update", applied=True, kind=kind, loss_before=loss_before,
            loss_after=loss_after, surprise=surprise, lr=lr, drift=drift,
            replayed=len(replay_examples), canary=report.canary,
            rolled_back=report.rolled_back, turns=list(turns),
        )
        return report

    def observe_text(self, text: str, weight: float = 1.0) -> UpdateReport:
        """Learn from a short passage in a single step (`/teach`).

        For anything longer than a paragraph upload it as a document, which
        covers the whole text and takes several passes over it."""
        example = self._text_example(text)
        if example is None:
            return UpdateReport(False, "too short", 0.0, 0.0, 0.0, self.lr)
        self.replay.add(["(document)", text], weight=weight, kind="document")
        self._remember(["(document)", text], "document")
        loss_before = self.sequence_loss(example)
        if self.ema_loss is None:
            self.ema_loss = loss_before
        lr = self.lr * max(0.1, weight)
        replay = self._replay_examples(self.cfg.replay_batch)
        grad_norm, drift = self._step([example] + replay, lr)
        report = UpdateReport(True, "learned passage", loss_before,
                              self.sequence_loss(example), loss_before - self.ema_loss,
                              lr, grad_norm, drift, len(replay))
        self._after_update(report)
        self.journal.write(event="update", applied=True, kind="teach",
                           loss_before=report.loss_before, loss_after=report.loss_after,
                           lr=lr, drift=drift, canary=report.canary,
                           rolled_back=report.rolled_back)
        return report

    # -- uploads ---------------------------------------------------------

    def iter_learn_units(
        self,
        make_units: Callable[[], Iterable[str]],
        passes: int | None = None,
        weight: float = 1.0,
        name: str = "document",
        should_stop: StopCheck = None,
    ) -> Generator[LearnProgress, None, UpdateReport]:
        """Learn a document's language and voice, streamed.

        `make_units` is called once per pass and must return a fresh stream
        of sentences (`aria.documents.iter_units`). Nothing is held but a
        small shuffle buffer, so the document can be any size. A sample of
        it is filed into the replay buffer at the end, so later
        conversations keep rehearsing it."""
        keep = _Reservoir(DOCUMENT_REPLAY_CHUNKS, self.rng)
        words = [0]

        def remembered(units: Iterable[str]) -> Iterator[str]:
            chunk = ""
            for u in units:
                words[0] += len(u.split())
                chunk = f"{chunk} {u}" if chunk else u
                if len(chunk) >= 600:
                    keep.add(chunk)
                    chunk = ""
                yield u
            if len(chunk) > 40:
                keep.add(chunk)

        def source(first_pass: bool):
            units = make_units()
            return self._units_examples(remembered(units) if first_pass else units)

        report = yield from self._iter_train(source, passes, weight, "document", name,
                                             should_stop)
        for text in keep.items:
            self.replay.add(["(document)", text], weight=weight, kind="document")
            self._remember(["(document)", text], "document")
        report.words = words[0]
        return report

    def iter_learn_dialogues(
        self,
        make_convos: Callable[[], Iterable[Sequence[str]]],
        passes: int | None = None,
        weight: float = 1.0,
        name: str = "transcript",
        should_stop: StopCheck = None,
    ) -> Generator[LearnProgress, None, UpdateReport]:
        """Learn how someone answers, from conversations in which they play
        Aria (`aria.documents.iter_dialogues`), streamed like a document.
        Every HOLDOUT_EVERY-th conversation is held out."""
        keep = _Reservoir(DOCUMENT_REPLAY_CHUNKS, self.rng)

        def source(first_pass: bool):
            for i, convo in enumerate(make_convos()):
                ex = self._encode_turns(convo)
                if ex is None:
                    continue
                if first_pass:
                    keep.add(list(convo))
                yield "dialogue", ex, i % HOLDOUT_EVERY == HOLDOUT_EVERY - 1

        report = yield from self._iter_train(source, passes, weight, "transcript", name,
                                             should_stop)
        for convo in keep.items:
            self.replay.add(convo, weight=weight, kind="transcript")
            self._remember(convo, "transcript")
        return report

    def learn_document(self, text: str, passes: int | None = None, weight: float = 1.0,
                       name: str = "document", progress: Progress = None,
                       should_stop: StopCheck = None) -> UpdateReport:
        """`iter_learn_units` for text already in memory, run to the end."""
        return drive(self.iter_learn_units(lambda: iter_units(text.splitlines()), passes,
                                           weight, name, should_stop), progress)

    def learn_dialogues(self, convos: Sequence[Sequence[str]], passes: int | None = None,
                        weight: float = 1.0, name: str = "transcript",
                        progress: Progress = None, should_stop: StopCheck = None
                        ) -> UpdateReport:
        return drive(self.iter_learn_dialogues(lambda: iter(convos), passes, weight, name,
                                               should_stop), progress)

    def _draw(self, buf: list[Example], n: int) -> list[Example]:
        """Remove `n` random examples from `buf` (swap-and-pop, O(n))."""
        out = []
        for _ in range(min(n, len(buf))):
            j = self.rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            out.append(buf.pop())
        return out

    def _iter_train(
        self,
        source: Callable[[bool], Iterable[tuple[str, Example, bool]]],
        passes: int | None,
        weight: float,
        kind: str,
        name: str,
        should_stop: StopCheck,
    ) -> Generator[LearnProgress, None, UpdateReport]:
        """Several shuffled passes over a streamed source, rehearsal mixed in.

        The first pass only reads: it counts examples (so progress has a
        total) and picks the held-out probe. Then each pass streams the
        source again. Examples of different kinds (windows vs. exchanges)
        are batched separately, so short ones aren't padded to long ones.
        """
        bs = max(1, self.cfg.document_batch)
        counts: dict[str, int] = {}
        probe = _Reservoir(PROBE_SIZE, self.rng)
        fallback = _Reservoir(PROBE_SIZE, self.rng)
        n_held = 0
        for i, (bucket, ex, held) in enumerate(source(True)):
            if held:
                probe.add(ex)
                n_held += 1
            else:
                counts[bucket] = counts.get(bucket, 0) + 1
                fallback.add(ex)
            if (i + 1) % 2000 == 0:
                yield LearnProgress("reading", examples=i + 1)
        n_train = sum(counts.values())
        if n_train == 0:
            reason = "too short" if kind == "document" else "no usable exchanges"
            return UpdateReport(False, reason, 0.0, 0.0, 0.0, self.lr)

        # A short document has no held-out text; its loss is then measured on
        # a sample of what it was trained on, and labelled as such.
        heldout = n_held > 0
        probe_set = probe.items if heldout else fallback.items
        steps_per_pass = sum(math.ceil(c / bs) for c in counts.values())
        passes = max(1, passes or self.cfg.document_passes)
        passes = max(passes, math.ceil(self.cfg.document_min_steps / steps_per_pass))
        total = passes * steps_per_pass
        if self.cfg.document_max_steps > 0:
            total = min(total, self.cfg.document_max_steps)
        lr = self.lr * max(0.1, weight) * self.cfg.document_lr_scale

        loss_before = self.mean_loss(probe_set)
        if self.ema_loss is None:
            self.ema_loss = loss_before
        report = UpdateReport(True, f"learned {kind}", loss_before, loss_before,
                              loss_before - self.ema_loss, lr, steps=0,
                              heldout=heldout, examples=n_train)

        def batches() -> Iterator[list[Example]]:
            for _ in range(passes):
                buffers: dict[str, list[Example]] = {}
                for bucket, ex, held in source(False):
                    if held:
                        continue
                    buf = buffers.setdefault(bucket, [])
                    buf.append(ex)
                    if len(buf) >= max(SHUFFLE_BUFFER, bs):
                        yield self._draw(buf, bs)
                for buf in buffers.values():
                    while buf:
                        yield self._draw(buf, bs)

        replayed = rollbacks = 0
        for batch in batches():
            if report.steps >= total:
                break
            if should_stop is not None and should_stop():
                report.stopped = True
                break
            # Fewer rehearsal samples than in chat: the document is the point,
            # and the batch already holds several pieces of it.
            replay = self._replay_examples(max(1, self.cfg.replay_batch // 2))
            replayed += len(replay)
            report.grad_norm, report.drift = self._step(batch + replay, lr)
            report.steps += 1

            step_report = UpdateReport(True, "", 0.0, 0.0, 0.0, lr)
            self._after_update(step_report)
            report.consolidated |= step_report.consolidated
            if step_report.canary is not None:
                report.canary = step_report.canary
            if step_report.rolled_back:
                # The canary undid the last stretch and halved the learning
                # rate. Carry on more gently; if it keeps happening, the
                # material itself is what's hurting her English.
                report.rolled_back = True
                rollbacks += 1
                lr *= self.cfg.rollback_lr_decay
                if rollbacks >= MAX_ROLLBACKS_PER_UPLOAD:
                    break
            yield LearnProgress("learning", report.steps, total, n_train)

        report.loss_after = self.mean_loss(probe_set)
        report.replayed = replayed
        self.journal.write(
            event="update", applied=True, kind=kind, name=name,
            examples=n_train, heldout=heldout, steps=report.steps,
            loss_before=report.loss_before, loss_after=report.loss_after,
            lr=lr, drift=report.drift, canary=report.canary,
            rolled_back=report.rolled_back, consolidated=report.consolidated,
            stopped=report.stopped,
        )
        return report

    # ------------------------------------------------------------------
    # safety net
    # ------------------------------------------------------------------

    def _health_check(self, report: UpdateReport) -> float:
        canary = self.canary_loss()
        limit = self.canary_baseline * (1.0 + self.cfg.health_tolerance)
        if canary > limit:
            self.consecutive_bad += 1
            if self.consecutive_bad >= self.cfg.health_patience:
                self.rollback(canary)
                report.rolled_back = True
                self.consecutive_bad = 0
        else:
            self.consecutive_bad = 0
            # A genuinely better model becomes the new bar to beat.
            if canary < self.canary_baseline:
                self.canary_baseline = canary
        return canary

    def _remember(self, turns: Sequence[str], kind: str) -> None:
        if self.hippocampus is not None:
            self.hippocampus.store(turns, kind)

    def sleep(self) -> None:
        """Reconsolidation: re-encode every hippocampal memory with the
        cortex as it is now, so memories keep matching as the cortex learns."""
        if self.hippocampus is not None:
            self.hippocampus.rebuild(self.replay.items)

    def rollback(self, canary: float) -> None:
        if not self.last_good:
            return
        self._restore(self.last_good)
        self.opt = self._new_optimizer(self.lr)
        self.lr *= self.cfg.rollback_lr_decay
        self.sleep()
        self.journal.write(event="rollback", canary=canary,
                           baseline=self.canary_baseline, new_lr=self.lr)

    def consolidate(self) -> bool:
        """Accept the current weights and make the learning permanent.

        Returns False when the canary says the model is not healthy enough."""
        canary = self.canary_loss()
        if (self.cfg.health_check
                and canary > self.canary_baseline * (1.0 + self.cfg.health_tolerance)):
            # Not healthy enough to consolidate; leave the snapshot alone.
            self.journal.write(event="consolidate", skipped=True, canary=canary)
            return False

        if self.cfg.plasticity == "lora":
            # Fold the adapters into the real weight matrices, then start fresh
            # adapters from zero. After this the knowledge lives in the model's
            # own weights and the trust region resets around the new position.
            merge_lora(self.model)
            self._configure_plasticity()
            self.opt = self._new_optimizer(self.lr)
        elif self.anchor:
            with torch.no_grad():
                named = dict(self.model.named_parameters())
                e = self.cfg.anchor_ema
                for n in self.trainable_names:
                    if n in self.anchor:
                        a = self.anchor[n]
                        a.copy_(a.float().mul_(e).add_(named[n].detach(), alpha=1 - e))

        if self.cfg.health_check:
            self.last_good = self._snapshot()
        self.canary_baseline = min(self.canary_baseline, canary)
        # Each rollback halved the learning rate; healthy consolidations win
        # it back, so a few bad turns can't leave her unable to learn forever.
        self.lr = min(self.cfg.learning_rate, self.lr * 1.5)
        self.sleep()
        self.journal.write(event="consolidate", skipped=False, canary=canary,
                           baseline=self.canary_baseline,
                           updates=self.updates_applied)
        return True

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _runtime_path(self) -> Path:
        return self.state_dir / "learner_state.json"

    def _load_runtime_state(self) -> None:
        p = self._runtime_path()
        if not p.exists():
            return
        d = json.loads(p.read_text())
        self.lr = d.get("lr", self.lr)
        self.ema_loss = d.get("ema_loss", self.ema_loss)
        self.updates_applied = d.get("updates_applied", 0)
        self.turns_seen = d.get("turns_seen", 0)
        self.since_health = d.get("since_health", 0)
        self.since_consolidation = d.get("since_consolidation", 0)
        self.tokens_learned = d.get("tokens_learned", 0)
        if "canary_baseline" in d:
            # The stored baseline belongs to the stored weights. Keep whichever
            # is lower so a fresh, better model is not held to a stale bar.
            self.canary_baseline = min(self.canary_baseline, d["canary_baseline"])

    def save(self, weights: bool = True) -> None:
        self.replay.save(self.state_dir / "replay.json")
        self._runtime_path().write_text(json.dumps({
            "lr": self.lr,
            "ema_loss": self.ema_loss,
            "updates_applied": self.updates_applied,
            "turns_seen": self.turns_seen,
            "since_health": self.since_health,
            "since_consolidation": self.since_consolidation,
            "tokens_learned": self.tokens_learned,
            "canary_baseline": self.canary_baseline,
            "plasticity": self.cfg.plasticity,
        }, indent=2))
        if weights:
            # The *merged* weights, so the file is a plain model checkpoint
            # that loads without knowing anything about LoRA; in float16 with
            # tied matrices stored once (see aria.storage); written atomically
            # so an interrupted save can't destroy what was learned.
            atomic_save(
                {
                    # On the CPU, so the file loads on any machine.
                    "model": {k: v.cpu() for k, v in
                              half_state_dict(merged_state_dict(self.model)).items()},
                    # Growth adds layers; the base checkpoint doesn't know.
                    "n_layer": self.model.cfg.n_layer,
                    "updates_applied": self.updates_applied,
                    "canary_baseline": self.canary_baseline,
                },
                self.state_dir / "learned.pt",
            )

    def capacity_tokens(self) -> float:
        """Roughly how much text this model has room to learn from.

        The rule of thumb from scaling-law studies is ~20 training tokens per
        parameter: past that point a larger model gets more out of the same
        text than more passes through a small one. It is a guide, not a wall
        — the model keeps learning, just less per token."""
        return self.cfg.tokens_per_param * self.model.num_params()

    def room_left(self) -> float:
        """Fraction of `capacity_tokens` not yet used (never below 0)."""
        used = self.prior_tokens + self.tokens_learned
        return max(0.0, 1.0 - used / self.capacity_tokens())

    def status(self) -> dict[str, Any]:
        return {
            "plasticity": self.cfg.plasticity,
            "trainable_tensors": len(self.trainable_names),
            "trainable_params": sum(p.numel() for p in self.trainable),
            "turns_seen": self.turns_seen,
            "updates_applied": self.updates_applied,
            "learning_rate": self.lr,
            "ema_loss": self.ema_loss,
            "canary_baseline": self.canary_baseline,
            "canary_now": self.canary_loss(),
            "replay_size": len(self.replay),
            "replay_seen": self.replay.seen,
            "disk_mb": round(dir_size_mb(self.state_dir), 2),
            "layers": self.model.cfg.n_layer,
            "tokens_learned": self.tokens_learned,
            "room_left": round(self.room_left(), 3),
            "memory_saver": self.saving_memory,
            "hippocampus_memories": len(self.hippocampus.episodes) if self.hippocampus else 0,
            "hippocampus_mb": round(self.hippocampus.memory_mb(), 2) if self.hippocampus else 0.0,
            "recalls": self.hippocampus.recalls if self.hippocampus else 0,
        }


def resume_learned_weights(model: GPT, state_dir: str | Path,
                           device: str = "cpu") -> bool:
    """Load `learned.pt` over a freshly-built base model, if it exists.

    Call this before constructing `OnlineLearner`, so everything learned in
    previous sessions is in place before new adapters are attached."""
    p = Path(state_dir) / "learned.pt"
    if not p.exists():
        return False
    # mmap: the file is paged in as needed, not copied whole into memory.
    # Read to the CPU (memory-mapped) and let load_state_dict copy onto the
    # device: mmap straight onto a GPU isn't supported everywhere (Apple's).
    state = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
    grown_to = state.get("n_layer", model.cfg.n_layer)
    if grown_to > model.cfg.n_layer:
        # She grew in an earlier session: give the base model the same
        # (still empty) layers so the learned weights have somewhere to go.
        from .model import grow
        grow(model, grown_to - model.cfg.n_layer)
    try:
        model.load_state_dict(state["model"])
    except RuntimeError as e:
        raise RuntimeError(
            f"{p} was learned on top of a different model than the one being "
            f"loaded. Point --state-dir somewhere else, or delete {p.parent} "
            f"to start over.\n{e}"
        ) from None
    return True


def bf16_supported(device: str) -> bool:
    try:
        if str(device).startswith("cuda"):
            return torch.cuda.is_bf16_supported()
        if str(device).startswith("mps"):
            # Metal's bfloat16 depends on the macOS version and is not
            # reliably faster; an Apple GPU stays in float32.
            return False
        return bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
    except (AttributeError, RuntimeError):
        return False


def drive(gen: Generator[LearnProgress, None, UpdateReport],
          progress: Progress = None) -> UpdateReport:
    """Run a learning generator to the end, reporting each step."""
    while True:
        try:
            p = next(gen)
        except StopIteration as done:
            return done.value
        if progress is not None:
            progress(p)
