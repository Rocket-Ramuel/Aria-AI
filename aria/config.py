"""Configuration objects for Aria.

Everything that defines a model or a training run lives here as a plain
dataclass so it can be serialised into a checkpoint and reloaded verbatim.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


# --------------------------------------------------------------------------
# Special tokens. The chat format is:
#
#   <|bos|><|user|> hello there <|eot|><|aria|> hi! <|eot|><|user|> ...
#
# The learner only ever computes loss on the tokens *after* an <|aria|> marker,
# so the model is trained to produce replies rather than to imitate the user.
# --------------------------------------------------------------------------
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eot|>", "<|user|>", "<|aria|>"]


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 4          # grouped-query attention; must divide n_head
    n_embd: int = 256
    block_size: int = 256       # context window in tokens
    ffn_mult: float = 8 / 3     # SwiGLU hidden size = ffn_mult * n_embd (rounded)
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Cortical areas (aria.model.Areas): split each feed-forward layer into
    # this many specialists, two used per word. 0 = one dense layer.
    n_areas: int = 0
    area_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} not divisible by n_head={self.n_head}")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head={self.n_head} not divisible by n_kv_head={self.n_kv_head}")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    """Hyper-parameters for the offline pretraining pass."""

    batch_size: int = 16
    grad_accum: int = 1
    max_steps: int = 4000
    learning_rate: float = 3e-4
    min_lr_frac: float = 0.1     # cosine decays to min_lr_frac * learning_rate
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_batches: int = 20
    checkpoint_interval: int = 500
    seed: int = 1337
    # Wall-clock budget. Training stops cleanly when exceeded, whichever comes
    # first with max_steps. 0 disables the limit.
    max_minutes: float = 0.0
    # Diagonal Fisher information is estimated at the end of pretraining and
    # stored in the checkpoint; the online learner uses it for EWC.
    fisher_batches: int = 64


@dataclass
class LearnerConfig:
    """Hyper-parameters for the *online* (always-on) learning loop.

    The defaults here are deliberately conservative. Online gradient descent on
    a live conversation is the fastest known route to a model that parrots its
    last input and has forgotten English, so every knob below exists to make
    that failure mode not happen.
    """

    # --- what gets updated -------------------------------------------------
    plasticity: Literal["full", "ffn", "lora"] = "lora"
    lora_rank: int = 8
    lora_alpha: float = 16.0

    # --- the update itself -------------------------------------------------
    learning_rate: float = 1e-4
    steps_per_turn: int = 1        # inner gradient steps per conversational turn
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.99

    # --- rehearsal ---------------------------------------------------------
    replay_batch: int = 6          # replayed sequences per new sequence
    pretrain_replay_frac: float = 0.5   # of those, share drawn from the pretrain corpus
    replay_capacity: int = 4096    # reservoir size for conversational memories

    # --- whose words are learned -------------------------------------------
    # The user's own messages are also learned as if Aria had said them, so
    # her replies drift toward how the person she talks to writes.
    style_mirror: bool = True
    style_weight: float = 0.5      # relative weight of the user's words vs. the reply
    # Training on Aria's own sampled reply mostly reinforces what she already
    # says. Harmless for a pretrained model; for a blank one it would teach her
    # her own babble, so the blank preset turns it off.
    learn_own_replies: bool = True

    # --- uploaded documents --------------------------------------------------
    document_passes: int = 3       # epochs over an uploaded document
    document_batch: int = 4        # document windows per gradient step
    document_min_steps: int = 24   # a short sample still gets a real lesson
    document_max_steps: int = 0    # 0 = no limit: every pass over every page
    # An upload is deliberate material, so it moves faster than chat. Measured
    # on the shipped model: 3x cuts loss on a 200-word sample from 6.4 to about 4
    # with the canary unchanged; 10x starts to cost general English.
    document_lr_scale: float = 3.0

    # --- regularisation toward the anchor ----------------------------------
    ewc_lambda: float = 250.0      # weight on Fisher-weighted pull to the anchor
    l2_anchor: float = 1e-3        # plain L2 pull to the anchor (backstop if no Fisher)
    trust_radius: float = 0.05     # max ||theta - anchor|| / ||anchor|| per tensor; 0 = off

    # --- surprise gating ---------------------------------------------------
    # Learn from what is novel, ignore what the model already predicts well.
    # Set surprise_gate False to take a step on literally every turn.
    surprise_gate: bool = True
    surprise_floor: float = 0.05   # skip update if loss < ema_loss * (1 + floor)
    surprise_ema: float = 0.98
    lr_surprise_scale: float = 1.0  # lr multiplier = 1 + scale * normalised surprise
    max_lr_scale: float = 3.0

    # --- safety net --------------------------------------------------------
    # The canary measures loss on general English. That is the right guard for
    # a pretrained model and the wrong one for a blank model learning one
    # person's idiolect, where drifting away from "general English" is the goal.
    health_check: bool = True
    health_interval: int = 8       # run the canary eval every N applied updates
    health_tolerance: float = 0.06  # allowed relative rise in canary loss
    health_patience: int = 2       # consecutive failures before rollback
    rollback_lr_decay: float = 0.5  # lr multiplier applied after a rollback

    # --- consolidation ("sleep") -------------------------------------------
    consolidate_interval: int = 32  # applied updates between consolidations
    anchor_ema: float = 0.9         # anchor <- ema*anchor + (1-ema)*current

    # --- memory and growth -----------------------------------------------
    # "auto" turns memory saving on above 20M parameters: a low-memory
    # optimiser (~2 bytes of state per weight instead of 8), activation
    # checkpointing, and half-precision safety snapshots.
    memory_saver: str = "auto"
    # Growth adds layers when the model has learned from more text than its
    # size has room for (tokens_per_param tokens per parameter), up to
    # grow_max_factor times its original depth. 0 turns automatic growth off;
    # /grow always works.
    tokens_per_param: float = 20.0
    grow_max_factor: float = 2.0

    # --- hippocampus (aria.hippocampus) -----------------------------------
    # Fast memory: words of recent experience kept for one-shot recall.
    # 16,384 words is ~8 MB for the small model; 0 turns it off.
    hippocampus_tokens: int = 16_384
    recall_threshold: float = 0.35   # how closely a moment must match
    recall_strength: float = 0.5     # most a memory can sway a prediction

    seed: int = 7


@dataclass
class AriaConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AriaConfig":
        return cls(
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            learner=LearnerConfig(**d.get("learner", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AriaConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Size presets. `tiny` exists so the test suite and a laptop CPU can run the
# whole pipeline end to end in minutes; `base` is what you'd reach for with a
# GPU and a few hours.
# --------------------------------------------------------------------------
PRESETS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(vocab_size=4096, n_layer=4, n_head=4, n_kv_head=2,
                        n_embd=128, block_size=128),
    "small": ModelConfig(vocab_size=8192, n_layer=6, n_head=8, n_kv_head=4,
                         n_embd=256, block_size=256),
    "base": ModelConfig(vocab_size=16384, n_layer=12, n_head=12, n_kv_head=4,
                        n_embd=768, block_size=512),
}


def preset(name: str) -> ModelConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return dataclasses.replace(PRESETS[name])


def blank_learner_config() -> LearnerConfig:
    """Learner settings for a model that starts from random weights.

    Every safeguard in the default config protects knowledge the model already
    has. A blank model has none, so the anchors, trust region, EWC, canary and
    corpus rehearsal would only stop it from learning. What remains is
    rehearsal of the user's own past words, which is what keeps one session
    from overwriting the last.
    """
    return LearnerConfig(
        plasticity="full",
        learning_rate=1e-3,
        steps_per_turn=2,
        replay_batch=4,
        pretrain_replay_frac=0.0,
        style_mirror=True,
        style_weight=1.0,
        learn_own_replies=False,
        document_passes=4,
        document_batch=8,
        document_lr_scale=1.0,
        ewc_lambda=0.0,
        l2_anchor=0.0,
        trust_radius=0.0,
        surprise_gate=False,
        health_check=False,
    )
