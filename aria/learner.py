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
   never drift more than `trust_radius` (relative) from its anchor. This caps
   the damage a single conversation — or a deliberately adversarial one — can do.

5. **A canary and rollback.** A fixed set of held-out English sentences is
   evaluated periodically. If loss on them rises past tolerance, the learner
   restores the last known-good snapshot and lowers its learning rate. Learning
   that makes the model worse is undone automatically.

6. **Consolidation ("sleep").** Periodically the current weights are accepted as
   the new known-good state and the anchor is moved toward them. In LoRA mode
   the adapters are folded into the base weight matrices at this point and reset
   to zero — which is what makes the learning permanent rather than a growing
   pile of adapters.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from .config import LearnerConfig
from .data import TokenStream, collate, encode_dialogue
from .memory import Journal, ReplayBuffer, canary_texts
from .model import GPT, IGNORE_INDEX, attach_lora, lora_parameters, merge_lora, LoRALinear
from .tokenizer import BPETokenizer

FFN_KEYS = ("gate_proj", "up_proj", "down_proj")


@dataclass
class UpdateReport:
    """What the learner did with one exchange."""

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

        self.opt = torch.optim.AdamW(
            self.trainable, lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2), weight_decay=0.0,
        )

        self.canaries = self._encode_canaries(canary_texts(extra_canaries))
        self.canary_baseline = self.canary_loss()
        self.lr = cfg.learning_rate
        self.ema_loss: float | None = None
        self.updates_applied = 0
        self.turns_seen = 0
        self.consecutive_bad = 0

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
            self.model.to(self.device)
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

        self.trainable_names = [n for n, p in self.model.named_parameters()
                                if p.requires_grad]
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not self.trainable:
            raise RuntimeError(f"plasticity={mode!r} left no trainable parameters")

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
    def sequence_loss(self, example: tuple[list[int], list[int]]) -> float:
        was_training = self.model.training
        self.model.eval()
        x, y = collate([example], self.tok.pad_id)
        _, loss, _ = self.model(x.to(self.device), y.to(self.device))
        if was_training:
            self.model.train()
        return float(loss)

    # ------------------------------------------------------------------
    # batch construction
    # ------------------------------------------------------------------

    def _encode_turns(self, turns: Sequence[str]):
        return encode_dialogue(self.tok, turns, self.model.cfg.block_size)

    def _replay_examples(self, n: int) -> list[tuple[list[int], list[int]]]:
        """Half from past conversations, half from the pretraining corpus.

        The corpus half is what actually anchors English; the conversation half
        is what stops older lessons from being overwritten by newer ones."""
        if n <= 0:
            return []
        n_pre = int(round(n * self.cfg.pretrain_replay_frac)) if self.pretrain_stream else 0
        n_chat = n - n_pre

        out: list[tuple[list[int], list[int]]] = []
        for item in self.replay.sample(n_chat):
            enc = self._encode_turns(item["turns"])
            if enc is not None:
                out.append(enc)
            if len(out) >= n_chat:
                break

        if n_pre > 0 and self.pretrain_stream is not None:
            xb, yb = self.pretrain_stream.batch(n_pre, self.torch_gen)
            out.extend([(x.tolist(), y.tolist()) for x, y in zip(xb, yb)])
        return out

    # ------------------------------------------------------------------
    # regularisation
    # ------------------------------------------------------------------

    def _anchor_penalty(self) -> torch.Tensor:
        named = dict(self.model.named_parameters())
        total = torch.zeros((), device=self.device)
        for n in self.trainable_names:
            p = named[n]
            if self.cfg.plasticity == "lora":
                # Anchor is the zero adapter: keep the correction small.
                delta_sq = p.pow(2).sum()
                total = total + self.cfg.l2_anchor * delta_sq
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

        Returns the largest relative drift observed (after projection)."""
        radius = self.cfg.trust_radius
        max_drift = 0.0
        named = dict(self.model.named_parameters())

        if self.cfg.plasticity == "lora":
            for module in self.model.modules():
                if not isinstance(module, LoRALinear):
                    continue
                base_norm = module.base.weight.norm()
                delta = (module.lora_B @ module.lora_A) * module.scaling
                dn = float(delta.norm())
                bn = float(base_norm) + 1e-12
                rel = dn / bn
                if rel > radius:
                    module.lora_B.mul_(radius / rel)
                    rel = radius
                max_drift = max(max_drift, rel)
            return max_drift

        for n in self.trainable_names:
            p = named[n]
            a = self.anchor[n]
            delta = p - a
            dn = float(delta.norm())
            an = float(a.norm()) + 1e-12
            rel = dn / an
            if rel > radius:
                p.copy_(a + delta * (radius / rel))
                rel = radius
            max_drift = max(max_drift, rel)
        return max_drift

    # ------------------------------------------------------------------
    # the update
    # ------------------------------------------------------------------

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
        example = self._encode_turns(turns)
        if example is None:
            return UpdateReport(False, "unencodable", 0.0, 0.0, 0.0, self.lr)

        loss_before = self.sequence_loss(example)
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
        for g in self.opt.param_groups:
            g["lr"] = lr

        self.model.train()
        replay_examples = self._replay_examples(self.cfg.replay_batch)
        grad_norm = 0.0

        for _ in range(max(1, self.cfg.steps_per_turn)):
            batch = [example] + replay_examples
            x, y = collate(batch, self.tok.pad_id)
            x = x[:, : self.model.cfg.block_size].to(self.device)
            y = y[:, : self.model.cfg.block_size].to(self.device)

            self.opt.zero_grad(set_to_none=True)
            _, ce, _ = self.model(x, y)
            loss = ce + self._anchor_penalty()
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.trainable, self.cfg.grad_clip)
            )
            self.opt.step()

        drift = self._project_trust_region()
        self.updates_applied += 1
        loss_after = self.sequence_loss(example)

        report = UpdateReport(
            applied=True, reason="learned", loss_before=loss_before,
            loss_after=loss_after, surprise=surprise, lr=lr,
            grad_norm=grad_norm, drift=drift, replayed=len(replay_examples),
        )

        if self.updates_applied % self.cfg.health_interval == 0:
            report.canary = self._health_check(report)
        if (self.updates_applied % self.cfg.consolidate_interval == 0
                and not report.rolled_back):
            self.consolidate()
            report.consolidated = True

        self.journal.write(
            event="update", applied=True, loss_before=loss_before,
            loss_after=loss_after, surprise=surprise, lr=lr, drift=drift,
            replayed=len(replay_examples), canary=report.canary,
            rolled_back=report.rolled_back, turns=list(turns),
        )
        return report

    def observe_text(self, text: str, weight: float = 1.0) -> UpdateReport:
        """Learn from a document rather than an exchange.

        The whole passage is treated as Aria's own output so gradient flows over
        all of it — this is how you teach her a body of text rather than a reply.
        """
        ids = [self.tok.bos_id, self.tok.aria_id]
        ids += self.tok.encode(" " + text.strip(), allowed_special=False)
        ids += [self.tok.eot_id]
        ids = ids[: self.model.cfg.block_size + 1]
        if len(ids) < 8:
            return UpdateReport(False, "too short", 0.0, 0.0, 0.0, self.lr)
        example = (ids[:-1], ids[1:])

        loss_before = self.sequence_loss(example)
        if self.ema_loss is None:
            self.ema_loss = loss_before
        self.replay.add(["(document)", text], weight=weight, kind="document")

        lr = self.lr * max(0.1, weight)
        for g in self.opt.param_groups:
            g["lr"] = lr

        self.model.train()
        replay_examples = self._replay_examples(self.cfg.replay_batch)
        x, y = collate([example] + replay_examples, self.tok.pad_id)
        x = x[:, : self.model.cfg.block_size].to(self.device)
        y = y[:, : self.model.cfg.block_size].to(self.device)
        self.opt.zero_grad(set_to_none=True)
        _, ce, _ = self.model(x, y)
        (ce + self._anchor_penalty()).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.trainable, self.cfg.grad_clip))
        self.opt.step()
        drift = self._project_trust_region()
        self.updates_applied += 1

        loss_after = self.sequence_loss(example)
        report = UpdateReport(True, "learned document", loss_before, loss_after,
                              loss_before - (self.ema_loss or loss_before), lr,
                              gn, drift, len(replay_examples))
        if self.updates_applied % self.cfg.health_interval == 0:
            report.canary = self._health_check(report)
        self.journal.write(event="update", applied=True, kind="document",
                           loss_before=loss_before, loss_after=loss_after,
                           lr=lr, drift=drift, canary=report.canary,
                           rolled_back=report.rolled_back)
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
        self.opt = torch.optim.AdamW(
            self.trainable, lr=self.lr,
            betas=(self.cfg.beta1, self.cfg.beta2), weight_decay=0.0,
        )
        self.lr *= self.cfg.rollback_lr_decay
        self.journal.write(event="rollback", canary=canary,
                           baseline=self.canary_baseline, new_lr=self.lr)

    def consolidate(self) -> None:
        """Accept the current weights and make the learning permanent."""
        canary = self.canary_loss()
        if canary > self.canary_baseline * (1.0 + self.cfg.health_tolerance):
            # Not healthy enough to consolidate; leave the snapshot alone.
            self.journal.write(event="consolidate", skipped=True, canary=canary)
            return

        if self.cfg.plasticity == "lora":
            # Fold the adapters into the real weight matrices, then start fresh
            # adapters from zero. After this the knowledge lives in the model's
            # own weights and the trust region resets around the new position.
            merge_lora(self.model)
            self._configure_plasticity()
            self.anchor = self._snapshot()
            self.opt = torch.optim.AdamW(
                self.trainable, lr=self.lr,
                betas=(self.cfg.beta1, self.cfg.beta2), weight_decay=0.0,
            )
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

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _runtime_path(self) -> Path:
        return self.state_dir / "learner_state.json"

    def _load_runtime_state(self) -> None:
        p = self._runtime_path()
        if not p.exists():
            return
        import json
        d = json.loads(p.read_text())
        self.lr = d.get("lr", self.lr)
        self.ema_loss = d.get("ema_loss", self.ema_loss)
        self.updates_applied = d.get("updates_applied", 0)
        self.turns_seen = d.get("turns_seen", 0)
        if "canary_baseline" in d:
            # The stored baseline belongs to the stored weights. Keep whichever
            # is lower so a fresh, better model is not held to a stale bar.
            self.canary_baseline = min(self.canary_baseline, d["canary_baseline"])

    def save(self, weights: bool = True) -> None:
        import json
        self.replay.save(self.state_dir / "replay.json")
        self._runtime_path().write_text(json.dumps({
            "lr": self.lr,
            "ema_loss": self.ema_loss,
            "updates_applied": self.updates_applied,
            "turns_seen": self.turns_seen,
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
    state = torch.load(p, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return True
