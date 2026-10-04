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
to. Uploaded documents and transcripts go through `learn_document` and
`learn_dialogues`, which take many steps over the material instead of one.

A model created blank (`aria.pretrain.create_blank_checkpoint`) runs with
`blank_learner_config()`, which switches off the safeguards that only exist to
protect pretrained knowledge.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch

from .config import LearnerConfig
from .data import TokenStream, collate, encode_dialogue
from .memory import Journal, ReplayBuffer, canary_texts
from .model import GPT, IGNORE_INDEX, attach_lora, lora_parameters, merge_lora, LoRALinear
from .tokenizer import BPETokenizer

FFN_KEYS = ("gate_proj", "up_proj", "down_proj")

Example = tuple[list[int], list[int]]
Progress = Optional[Callable[[int, int, float], None]]

# How many chunks of one uploaded document are kept for later rehearsal. Enough
# to keep the lesson alive, few enough that one book can't evict every
# conversation from the reservoir.
DOCUMENT_REPLAY_CHUNKS = 48

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


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
    drift: float = 0.0
    replayed: int = 0
    rolled_back: bool = False
    consolidated: bool = False
    canary: Optional[float] = None
    steps: int = 1

    def line(self) -> str:
        if not self.applied:
            return f"[learn] skipped ({self.reason}); loss {self.loss_before:.3f}"
        bits = [
            f"[learn] loss {self.loss_before:.3f} -> {self.loss_after:.3f}",
            f"surprise {self.surprise:+.2f}",
            f"lr {self.lr:.1e}",
            f"replay {self.replayed}",
            f"drift {self.drift:.3f}",
        ]
        if self.steps > 1:
            bits.insert(1, f"{self.steps} steps")
        if self.canary is not None:
            bits.append(f"canary {self.canary:.3f}")
        if self.consolidated:
            bits.append("consolidated")
        if self.rolled_back:
            bits.append("ROLLED BACK")
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
    ) -> None:
        self.model = model
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
        self.fisher = self._restrict_fisher(fisher)
        self.anchor = self._snapshot()
        self.last_good = self._snapshot()

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

        self._load_runtime_state()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _configure_plasticity(self) -> None:
        mode = self.cfg.plasticity
        if mode == "lora":
            n = attach_lora(self.model, self.cfg.lora_rank, self.cfg.lora_alpha)
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
        # New adapters are created on the CPU; every mode re-asserts the device
        # so a consolidation never leaves part of the model behind.
        self.model.to(self.device)

        self.trainable_names = [n for n, p in self.model.named_parameters()
                                if p.requires_grad]
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not self.trainable:
            raise RuntimeError(f"plasticity={mode!r} left no trainable parameters")

    def _new_optimizer(self, lr: float) -> torch.optim.AdamW:
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

    def _snapshot(self) -> dict[str, torch.Tensor]:
        named = dict(self.model.named_parameters())
        return {n: named[n].detach().clone() for n in self.trainable_names}

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
    def mean_loss(self, examples: Sequence[Example]) -> float:
        """Token-averaged loss over a few examples, evaluated in one batch."""
        was_training = self.model.training
        self.model.eval()
        x, y = self._batch(examples)
        _, loss, _ = self.model(x, y)
        if was_training:
            self.model.train()
        return float(loss)

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

    def _document_windows(self, text: str) -> list[Example]:
        """Cut a long text into overlapping context-sized training windows."""
        block = self.model.cfg.block_size
        ids = self.tok.encode(text.strip(), allowed_special=False)
        span = block - 2                    # leaves room for <bos><aria>
        stride = max(1, (span * 3) // 4)
        windows: list[Example] = []
        for start in range(0, max(1, len(ids) - span // 4), stride):
            chunk = ids[start : start + span]
            if len(chunk) < 6:
                break
            seq = [self.tok.bos_id, self.tok.aria_id] + chunk
            if start + span >= len(ids):
                seq.append(self.tok.eot_id)
            x, y = seq[:-1], seq[1:]
            y[0] = IGNORE_INDEX
            windows.append((x, y))
        return windows

    def _document_exchanges(self, text: str, limit: int = 2000) -> list[Example]:
        """The document as a run of exchanges: each sentence answers the last.

        Windows teach Aria to *continue* text in a voice; replies are produced
        after `<user> ... <eot><aria>`, a context windows never show. Pairing
        consecutive sentences puts the same voice in that position, so it is
        learned as a way of answering. For a model with no pretraining this is
        the difference between memorising a sample and replying in its style.
        """
        units = [u.strip() for line in text.splitlines()
                 for u in _SENTENCE_END.split(line) if len(u.strip()) > 1]
        out: list[Example] = []
        for a, b in zip(units, units[1:]):
            enc = self._encode_turns([a, b])
            if enc is not None:
                out.append(enc)
            if len(out) >= limit:
                break
        return out

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
                # Anchor is the zero adapter: keep the correction small.
                total = total + self.cfg.l2_anchor * p.pow(2).sum()
                continue
            delta = p - self.anchor[n]
            total = total + self.cfg.l2_anchor * delta.pow(2).sum()
            f = self.fisher.get(n)
            if f is not None:
                total = total + self.cfg.ewc_lambda * (f * delta.pow(2)).sum()
        return total

    @torch.no_grad()
    def _project_trust_region(self) -> float:
        """Clip each tensor back inside its allowed distance from the anchor.

        Returns the largest relative drift observed (after projection). A
        `trust_radius` of 0 disables the projection but still reports drift."""
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

        for n in self.trainable_names:
            p = named[n]
            a = self.anchor[n]
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

    def _step(self, batch: Sequence[Example], lr: float) -> tuple[float, float]:
        """One optimiser step on `batch`, then the trust-region projection."""
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.model.train()
        x, y = self._batch(batch)
        self.opt.zero_grad(set_to_none=True)
        _, ce, _ = self.model(x, y)
        (ce + self._anchor_penalty()).backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.trainable, self.cfg.grad_clip))
        self.opt.step()
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

        For anything longer than a paragraph use `learn_document`, which
        covers the whole text and takes several passes over it."""
        example = self._text_example(text)
        if example is None:
            return UpdateReport(False, "too short", 0.0, 0.0, 0.0, self.lr)
        self.replay.add(["(document)", text], weight=weight, kind="document")
        return self._train([example], passes=1, weight=weight, kind="teach")

    def learn_document(
        self,
        text: str,
        passes: int | None = None,
        weight: float = 1.0,
        name: str = "document",
        progress: Progress = None,
    ) -> UpdateReport:
        """Learn the voice of a whole document: several passes over it.

        The text is cut into overlapping windows that fill the context, and
        each window is learned as if Aria had written it. A sample of the
        document is filed into the replay buffer so later conversations keep
        rehearsing it."""
        windows = self._document_windows(text)
        if not windows:
            return UpdateReport(False, "too short", 0.0, 0.0, 0.0, self.lr)
        self._remember_document(text, weight)
        examples = windows + self._document_exchanges(text)
        return self._train(examples, passes, weight, kind="document", name=name,
                           progress=progress)

    def learn_dialogues(
        self,
        convos: Sequence[Sequence[str]],
        passes: int | None = None,
        weight: float = 1.0,
        name: str = "transcript",
        progress: Progress = None,
    ) -> UpdateReport:
        """Learn how someone answers, from conversations in which they play Aria.

        Each conversation alternates user/aria and ends on the line to learn
        (see `aria.documents.transcript_dialogues`)."""
        examples = [e for e in (self._encode_turns(c) for c in convos) if e is not None]
        if not examples:
            return UpdateReport(False, "no usable exchanges", 0.0, 0.0, 0.0, self.lr)
        keep = self.rng.sample(list(convos), min(len(convos), DOCUMENT_REPLAY_CHUNKS))
        for c in keep:
            self.replay.add(list(c), weight=weight, kind="transcript")
        return self._train(examples, passes, weight, kind="transcript", name=name,
                           progress=progress)

    def _remember_document(self, text: str, weight: float) -> None:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        if len(paragraphs) < 4:
            # One long block: fall back to fixed-size pieces.
            flat = " ".join(text.split())
            paragraphs = [flat[i : i + 600] for i in range(0, len(flat), 600)]
        if len(paragraphs) > DOCUMENT_REPLAY_CHUNKS:
            paragraphs = self.rng.sample(paragraphs, DOCUMENT_REPLAY_CHUNKS)
        for p in paragraphs:
            self.replay.add(["(document)", p], weight=weight, kind="document")

    def _train(
        self,
        examples: list[Example],
        passes: int | None,
        weight: float,
        kind: str,
        name: str = "",
        progress: Progress = None,
    ) -> UpdateReport:
        """Several shuffled passes over `examples`, with rehearsal mixed in."""
        passes = max(1, passes or self.cfg.document_passes)
        bs = max(1, self.cfg.document_batch)
        steps_per_pass = math.ceil(len(examples) / bs)
        total = passes * steps_per_pass
        lr = self.lr * max(0.1, weight)
        if kind in ("document", "transcript"):
            total = max(total, self.cfg.document_min_steps)
            lr *= self.cfg.document_lr_scale
        total = min(total, max(1, self.cfg.document_max_steps))
        probe = examples if len(examples) <= 16 else self.rng.sample(examples, 16)

        loss_before = self.mean_loss(probe)
        if self.ema_loss is None:
            self.ema_loss = loss_before

        report = UpdateReport(True, f"learned {kind}", loss_before, loss_before,
                              loss_before - self.ema_loss, lr, steps=0)
        order: list[int] = []
        replayed = 0
        for step in range(total):
            if not order:
                order = list(range(len(examples)))
                self.rng.shuffle(order)
            batch = [examples[order.pop()] for _ in range(min(bs, len(order)))]
            # Fewer rehearsal samples than in chat: the document is the point,
            # and the batch already holds several windows of it.
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
                # The safety net judged this material harmful; stop rather
                # than keep pushing the model somewhere it just rolled back from.
                report.rolled_back = True
                break
            if progress is not None:
                progress(step + 1, total, lr)

        report.loss_after = self.mean_loss(probe)
        report.replayed = replayed
        self.journal.write(
            event="update", applied=True, kind=kind, name=name,
            examples=len(examples), steps=report.steps,
            loss_before=report.loss_before, loss_after=report.loss_after,
            lr=lr, drift=report.drift, canary=report.canary,
            rolled_back=report.rolled_back, consolidated=report.consolidated,
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

    def rollback(self, canary: float) -> None:
        self._restore(self.last_good)
        self.opt = self._new_optimizer(self.lr)
        self.lr *= self.cfg.rollback_lr_decay
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
            self.anchor = self._snapshot()
            self.opt = self._new_optimizer(self.lr)
        else:
            with torch.no_grad():
                named = dict(self.model.named_parameters())
                e = self.cfg.anchor_ema
                for n in self.trainable_names:
                    self.anchor[n].mul_(e).add_(named[n].detach(), alpha=1 - e)

        self.last_good = self._snapshot()
        self.canary_baseline = min(self.canary_baseline, canary)
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
            "canary_baseline": self.canary_baseline,
            "plasticity": self.cfg.plasticity,
        }, indent=2))
        if weights:
            # Save the *merged* weights so the file is a plain model checkpoint
            # that can be loaded without knowing anything about LoRA.
            export = copy.deepcopy(self.model)
            merge_lora(export)
            torch.save(
                {
                    "model": export.state_dict(),
                    "updates_applied": self.updates_applied,
                    "canary_baseline": self.canary_baseline,
                },
                self.state_dir / "learned.pt",
            )

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
        }


def resume_learned_weights(model: GPT, state_dir: str | Path,
                           device: str = "cpu") -> bool:
    """Load `learned.pt` over a freshly-built base model, if it exists.

    Call this before constructing `OnlineLearner`, so everything learned in
    previous sessions is in place before new adapters are attached."""
    p = Path(state_dir) / "learned.pt"
    if not p.exists():
        return False
    state = torch.load(p, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state["model"])
    except RuntimeError as e:
        raise RuntimeError(
            f"{p} was learned on top of a different model than the one being "
            f"loaded. Point --state-dir somewhere else, or delete {p.parent} "
            f"to start over.\n{e}"
        ) from None
    return True
