"""The language model itself: a decoder-only transformer written from scratch.

Architecture notes (all standard, all implemented here rather than imported):

* pre-norm residual blocks with RMSNorm
* rotary position embeddings (RoPE) applied to queries and keys
* grouped-query attention, so the KV cache stays small during generation
* SwiGLU feed-forward
* tied input/output embeddings

Nothing in this file is pretrained or downloaded; `GPT(...)` starts from random
weights and everything it knows comes from `aria.pretrain` and `aria.learner`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

IGNORE_INDEX = -100


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device=None, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)              # (T, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim); cos/sin: (T, head_dim/2)."""
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # `is_causal=True` aligns its mask to the top-left, which is only what
        # we want when the queries *are* the whole sequence. With a non-empty
        # cache the queries sit at the end, so the mask has to be built
        # explicitly against absolute positions — otherwise a chunked prefill
        # silently attends to the wrong keys.
        dropout_p = self.dropout if self.training else 0.0
        if kv_cache is None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                               is_causal=True)
        elif T == 1:
            # A single query legitimately attends to every cached key.
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            kv_len = k.shape[2]
            q_pos = torch.arange(kv_len - T, kv_len, device=x.device).unsqueeze(1)
            k_pos = torch.arange(kv_len, device=x.device).unsqueeze(0)
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                               attn_mask=q_pos >= k_pos)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_drop(self.o_proj(y)), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.n_embd)
        hidden = 32 * ((hidden + 31) // 32)   # round up for friendlier matmuls
        self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale down the projections that write into the residual stream, so
        # residual variance does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        loss_reduction: str = "mean",
    ):
        """`targets` may contain IGNORE_INDEX at positions that should not
        contribute to the loss — that is how the chat format trains only on
        Aria's own tokens."""
        B, T = idx.shape
        pos_start = 0
        if kv_caches is not None and kv_caches[0] is not None:
            pos_start = kv_caches[0][0].shape[2]
        if pos_start + T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {pos_start + T} exceeds block_size {self.cfg.block_size}"
            )

        cos = self.rope_cos[pos_start : pos_start + T]
        sin = self.rope_sin[pos_start : pos_start + T]

        x = self.drop(self.tok_emb(idx))
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, c = block(x, cos, sin, cache)
            new_caches.append(c)
        x = self.norm_f(x)

        if targets is None:
            # Only the last position is needed to sample the next token.
            logits = self.lm_head(x[:, -1:, :])
            return logits, None, new_caches

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction=loss_reduction,
        )
        return logits, loss, new_caches

    def empty_cache(self) -> list[None]:
        return [None] * self.cfg.n_layer


# ---------------------------------------------------------------------------
# LoRA. Used by the online learner's default "lora" plasticity mode: the frozen
# base weights stay exactly as pretraining left them and only the low-rank
# adapters move, which bounds how far a conversation can shift the model.
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.lora_A, std=1.0 / rank)
        # B starts at zero, so an untrained adapter is an exact no-op.
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling


DEFAULT_LORA_TARGETS = ("q_proj", "v_proj", "o_proj", "down_proj")


def attach_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
) -> int:
    """Wrap the targeted `nn.Linear` layers in LoRA adapters, in place.

    Returns the number of layers adapted."""
    n = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha))
                n += 1
    return n


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if "lora_" in name]


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Fold every adapter back into its base weight and remove the wrapper.

    This is what makes online learning *permanent*: after merging, the knowledge
    lives in the ordinary weight matrices and the adapters restart from zero."""
    n = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight.copy_(child.merged_weight())
                for p in child.base.parameters():
                    p.requires_grad_(True)
                setattr(parent, child_name, child.base)
                n += 1
    return n
