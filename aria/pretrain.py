"""Offline pretraining: teach the model English before anyone talks to it.

This produces `runs/<name>/base.pt`, which contains the weights, the config, the
tokenizer, and the diagonal Fisher information used later by the online learner
to decide which weights are safe to move.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from pathlib import Path

import torch

from .config import AriaConfig, ModelConfig, TrainConfig, blank_learner_config, preset
from .data import ChatSet, TokenStream, mixed_batch
from .model import GPT
from .storage import (atomic_save, decode_fisher, dequantize_state_dict, encode_fisher,
                      half_state_dict, int8_state_dict)
from .tokenizer import BPETokenizer


def lr_at(step: int, cfg: TrainConfig, horizon: int | None = None) -> float:
    """Linear warmup, then cosine decay that reaches its floor at `horizon`.

    `horizon` defaults to `max_steps`. Under a wall-clock budget the run ends
    long before that, so the caller passes its projected final step instead;
    otherwise the learning rate would still be near its peak when time ran out.
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    end = min(cfg.max_steps, horizon or cfg.max_steps)
    progress = (step - cfg.warmup_steps) / max(1, end - cfg.warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    min_lr = cfg.learning_rate * cfg.min_lr_frac
    return min_lr + coeff * (cfg.learning_rate - min_lr)


def build_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    # Weight-decay only the matrices; norms and embeddings are left alone.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )


@torch.no_grad()
def evaluate(model: GPT, stream: TokenStream, batches: int, batch_size: int,
             generator: torch.Generator) -> float:
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    for _ in range(batches):
        x, y = stream.batch(batch_size, generator)
        _, loss, _ = model(x.to(device), y.to(device))
        total += float(loss)
    model.train()
    return total / max(1, batches)


def estimate_fisher(
    model: GPT,
    stream: TokenStream,
    batches: int,
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Diagonal empirical Fisher: E[(dL/dtheta)^2] over the pretraining data.

    Large entries mark weights the pretrained knowledge is sensitive to. The
    online learner uses this to pull those weights back hard while leaving the
    insensitive ones free to move.

    The expectation is over *individual sequences*: each one gets its own
    backward pass and its own squared gradient. Squaring the gradient of a
    batch-mean loss instead would give (E[g])², which cancels wherever
    sequences disagree and underestimates exactly the weights that matter.
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
              if p.requires_grad}
    device = next(model.parameters()).device
    n_samples = 0
    for i in range(batches):
        x, y = stream.batch(batch_size, generator)
        for j in range(x.shape[0]):
            model.zero_grad(set_to_none=True)
            _, loss, _ = model(x[j : j + 1].to(device), y[j : j + 1].to(device))
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach() ** 2
            n_samples += 1
    model.zero_grad(set_to_none=True)
    for n in fisher:
        fisher[n] /= max(1, n_samples)
    model.train()
    return fisher


def save_checkpoint(path: Path, model: GPT, cfg: AriaConfig, tok: BPETokenizer,
                    step: int, val_loss: float,
                    fisher: dict[str, torch.Tensor] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "tokenizer": {"specials": tok.specials, "merges": tok.merges},
            "step": step,
            "val_loss": val_loss,
            "fisher": fisher,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu"):
    # weights_only: checkpoints get shared, and a full unpickle of a file
    # someone sent you can run arbitrary code. Everything Aria stores is
    # tensors and plain containers, which the restricted loader handles.
    # mmap: weights are paged in from the file as they are copied into the
    # model, instead of being read into memory first — one copy, not two.
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    cfg = AriaConfig.from_dict(ckpt["config"])
    tok = BPETokenizer(merges=[tuple(m) for m in ckpt["tokenizer"]["merges"]],
                       specials=ckpt["tokenizer"]["specials"])
    # However the Fisher was stored (float32, float16, 8-bit log codes),
    # callers get float32.
    ckpt["fisher"] = decode_fisher(ckpt.get("fisher"))
    ckpt["model"] = dequantize_state_dict(ckpt["model"])
    model = GPT(cfg.model).to(device)
    # load_state_dict casts on copy, so a half-precision export loads straight
    # into the float32 model without any special handling here.
    model.load_state_dict(ckpt["model"])
    return model, tok, cfg, ckpt


# Where `--checkpoint` looks when it is not given explicitly: a model you
# trained yourself wins, then whatever ships with the repo.
DEFAULT_CHECKPOINTS = ("runs/aria/base.pt", "checkpoints/aria-small.pt")


def resolve_checkpoint(path: str | Path | None) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no checkpoint at {p}")
        return p
    for candidate in DEFAULT_CHECKPOINTS:
        if Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "no checkpoint found. Train one with `aria quickstart`, or pass "
        "--checkpoint explicitly. Looked in: " + ", ".join(DEFAULT_CHECKPOINTS)
    )


def export_checkpoint(src: str | Path, dst: str | Path, half: bool = True,
                      keep_fisher: bool = True,
                      learned: str | Path | None = None,
                      int8: bool = False) -> dict:
    """Write a compact, shareable copy of a checkpoint.

    Half precision halves the file for no measurable quality cost at this size
    (the weights are loaded back into a float32 model); the tied embedding is
    stored once; the Fisher information takes one byte per weight (see
    `aria.storage`). Together that is the difference between a checkpoint that
    is reasonable to commit and one that is not.

    With `learned` (a `learned.pt`), the weights are the ones Aria has learned
    since, so everything she knows travels as one self-contained file.
    """
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    weights = dequantize_state_dict(ckpt["model"])
    config = ckpt["config"]
    if learned is not None:
        state = torch.load(learned, map_location="cpu", weights_only=True)
        weights = state["model"]
        if "n_layer" in state:            # she may have grown since
            config = {**config, "model": {**config["model"], "n_layer": state["n_layer"]}}
    fisher = decode_fisher(ckpt.get("fisher")) if keep_fisher else None

    out = {
        "model": int8_state_dict(weights) if int8 else half_state_dict(weights) if half else
                 {k: v.float() if v.is_floating_point() else v for k, v in weights.items()},
        "config": config,
        "tokenizer": ckpt["tokenizer"],
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "fisher": (encode_fisher(fisher) if half else fisher) if fisher else None,
    }
    dst = Path(dst)
    atomic_save(out, dst)
    return {
        "int8": int8,
        "source_mb": Path(src).stat().st_size / 1e6,
        "export_mb": dst.stat().st_size / 1e6,
        "half": half,
        "fisher": out["fisher"] is not None,
        "step": out["step"],
        "val_loss": out["val_loss"],
    }


def pretrain(
    data_dir: str | Path = "data",
    out_dir: str | Path = "runs/aria",
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    chat_frac: float = 0.25,
    resume: bool = True,
    device: str = "auto",
    verbose: bool = True,
) -> Path:
    from .device import describe, resolve
    device = resolve(device)
    if verbose:
        print(f"training on the {describe(device)}", flush=True)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    model_cfg = model_cfg or ModelConfig()
    model_cfg.vocab_size = tok.vocab_size
    train_cfg = train_cfg or TrainConfig()
    cfg = AriaConfig(model=model_cfg, train=train_cfg)

    torch.manual_seed(train_cfg.seed)
    rng = random.Random(train_cfg.seed)
    gen = torch.Generator().manual_seed(train_cfg.seed)

    train_stream = TokenStream(data_dir / "train.bin", model_cfg.block_size)
    val_stream = TokenStream(data_dir / "val.bin", model_cfg.block_size)
    chatset = ChatSet(data_dir / "chat.pt", tok.pad_id) if (data_dir / "chat.pt").exists() else None

    model = GPT(model_cfg).to(device)
    opt = build_optimizer(model, train_cfg)
    start_step = 0

    ckpt_path = out_dir / "base.pt"
    latest = out_dir / "latest.pt"
    if resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        saved = state.get("config", {}).get("model")
        if saved is not None and saved != dataclasses.asdict(model_cfg):
            diff = {k: (saved.get(k), v) for k, v in dataclasses.asdict(model_cfg).items()
                    if saved.get(k) != v}
            raise ValueError(
                f"{latest} was trained with a different model shape "
                f"(saved vs requested: {diff}). Use the same --preset, pass "
                f"--no-resume to start over, or choose another --out-dir."
            )
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            opt.load_state_dict(state["optimizer"])
        start_step = state.get("step", 0)
        if verbose:
            print(f"resumed from {latest} at step {start_step}", flush=True)

    if verbose:
        print(f"model: {model.num_params()/1e6:.2f}M params "
              f"({model.num_params(non_embedding=True)/1e6:.2f}M non-embedding)", flush=True)
        print(f"data: {len(train_stream)/1e6:.2f}M train tokens, "
              f"{len(chatset) if chatset else 0} chat examples", flush=True)

    model.train()
    t0 = time.time()
    log_path = out_dir / "train_log.jsonl"
    best_val = float("inf")
    step = start_step
    stop_reason = "max_steps"

    horizon = None
    while step < train_cfg.max_steps:
        lr = lr_at(step, train_cfg, horizon)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = mixed_batch(train_stream, chatset, train_cfg.batch_size,
                               chat_frac, rng, gen, tok.pad_id,
                               block_size=model_cfg.block_size)
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y)
            (loss / train_cfg.grad_accum).backward()
            total_loss += loss.item() / train_cfg.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        step += 1

        elapsed = (time.time() - t0) / 60.0
        if train_cfg.max_minutes and step - start_step >= 10 and elapsed > 0:
            # Re-project where the time budget will end, so the cosine lands
            # on its floor when time runs out rather than at max_steps.
            rate = (step - start_step) / elapsed
            horizon = int(step + rate * max(0.0, train_cfg.max_minutes - elapsed))
        if verbose and (step % 10 == 0 or step == 1):
            print(f"step {step}/{train_cfg.max_steps}  loss {total_loss:.4f}  "
                  f"lr {lr:.2e}  |g| {float(grad_norm):.2f}  {elapsed:.1f}m", flush=True)

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_steps:
            val = evaluate(model, val_stream, train_cfg.eval_batches,
                           train_cfg.batch_size, gen)
            best_val = min(best_val, val)
            if verbose:
                print(f"  eval @ {step}: val loss {val:.4f}  (ppl {math.exp(min(val, 20)):.1f})",
                      flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "train_loss": total_loss,
                                    "val_loss": val, "lr": lr,
                                    "minutes": elapsed}) + "\n")

        if step % train_cfg.checkpoint_interval == 0:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "step": step, "config": cfg.to_dict()}, latest)

        if train_cfg.max_minutes and elapsed >= train_cfg.max_minutes:
            stop_reason = "time_budget"
            if verbose:
                print(f"stopping at step {step}: hit the {train_cfg.max_minutes}m budget",
                      flush=True)
            break

    final_val = evaluate(model, val_stream, train_cfg.eval_batches,
                         train_cfg.batch_size, gen)
    if verbose:
        print(f"estimating Fisher information ({train_cfg.fisher_batches} batches) ...",
              flush=True)
    fisher = estimate_fisher(model, train_stream, train_cfg.fisher_batches,
                             max(2, train_cfg.batch_size // 2), gen)

    save_checkpoint(ckpt_path, model, cfg, tok, step, final_val, fisher)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "config": cfg.to_dict()}, latest)
    if verbose:
        print(f"saved {ckpt_path}  (step {step}, val loss {final_val:.4f}, "
              f"stopped on {stop_reason})", flush=True)
    return ckpt_path


BLANK_CHECKPOINT = Path("runs/blank/base.pt")


def create_blank_checkpoint(path: str | Path = BLANK_CHECKPOINT,
                            size: str = "small", block_size: int = 512,
                            seed: int = 1337, areas: int = 0,
                            area_scale: float = 1.0) -> Path:
    """A model that knows nothing: random weights, no vocabulary, no grammar.

    The tokenizer has no merges, so it reads raw bytes — any language, any
    spelling, nothing assumed about English. Everything this model ever says it
    will have learned from the documents and messages it is given, which is
    the point: its voice can only be the voice of the people it learns from.

    The cost is that it says nothing coherent until it has read a fair amount.
    Expect babble for the first few thousand words of text, recognisable
    fragments of the source after a few tens of thousands.
    """
    path = Path(path)
    tok = BPETokenizer(merges=[])
    model_cfg = preset(size)
    model_cfg.vocab_size = tok.vocab_size
    # Bytes are short tokens; a longer window keeps a sentence or two in view.
    model_cfg.block_size = block_size
    model_cfg.n_areas, model_cfg.area_scale = areas, area_scale
    cfg = AriaConfig(model=model_cfg, learner=blank_learner_config())
    torch.manual_seed(seed)
    model = GPT(model_cfg)
    atomic_save({
        "model": half_state_dict(model.state_dict()),
        "config": cfg.to_dict(),
        "tokenizer": {"specials": tok.specials, "merges": tok.merges},
        "step": 0,
        "val_loss": float("nan"),
        "fisher": None,
    }, path)
    return path
