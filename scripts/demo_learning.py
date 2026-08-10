#!/usr/bin/env python3
"""An experiment, not a demo: measure whether the online learner actually works.

Four claims are checked against the real pretrained checkpoint, and each prints
the numbers it was judged on:

  1. ACQUISITION  — loss on a novel exchange falls when Aria is taught it.
  2. RETENTION    — that lesson survives learning 40 unrelated new things.
  3. STABILITY    — English (held-out canary sentences) does not degrade.
  4. PERSISTENCE  — the learning is still there after saving, exiting, and
                    reloading from a plain checkpoint in a fresh process image.

Run after `prepare` and `pretrain`:

    python scripts/demo_learning.py --checkpoint runs/aria/base.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aria.config import LearnerConfig
from aria.data import TokenStream, encode_dialogue
from aria.learner import OnlineLearner, resume_learned_weights
from aria.model import GPT
from aria.pretrain import load_checkpoint

LESSON = [
    "who wrote the notes in the blue folder",
    "Priya wrote the notes in the blue folder last Tuesday",
]

DISTRACTORS = [
    ["what is the harbour like", "the harbour is quiet and grey in the winter"],
    ["tell me about the bridge", "the bridge was rebuilt after the flood"],
    ["describe the market", "the market opens early and smells of bread"],
    ["what about the library", "the library keeps its older books downstairs"],
    ["how is the weather", "the weather has been cold but bright all week"],
]


def rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def build(checkpoint: Path, state_dir: Path, data_dir: Path, plasticity: str,
          device: str):
    model, tok, cfg, ckpt = load_checkpoint(checkpoint, device)
    resume_learned_weights(model, state_dir, device)
    stream = None
    if (data_dir / "train.bin").exists():
        stream = TokenStream(data_dir / "train.bin", model.cfg.block_size)
    lcfg = LearnerConfig(plasticity=plasticity, surprise_gate=False,
                         consolidate_interval=25)
    learner = OnlineLearner(model, tok, lcfg, fisher=ckpt.get("fisher"),
                            pretrain_stream=stream, state_dir=state_dir,
                            device=device)
    return model, tok, learner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/aria/base.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--state-dir", default="runs/demo_online")
    ap.add_argument("--plasticity", default="lora",
                    choices=["lora", "ffn", "full"])
    ap.add_argument("--repeats", type=int, default=12,
                    help="teaching passes over the lesson")
    ap.add_argument("--distractor-rounds", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep-state", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    state_dir = Path(args.state_dir)
    if state_dir.exists() and not args.keep_state:
        shutil.rmtree(state_dir)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"no checkpoint at {checkpoint}; run `python -m aria pretrain` first")
        return 1

    data_dir = Path(args.data_dir)
    model, tok, learner = build(checkpoint, state_dir, data_dir,
                                args.plasticity, args.device)
    block = model.cfg.block_size
    lesson_ex = encode_dialogue(tok, LESSON, block)

    print(f"model      {model.num_params()/1e6:.2f}M parameters")
    print(f"plasticity {args.plasticity} "
          f"({sum(p.numel() for p in learner.trainable)/1e3:.1f}K trainable)")

    # ---------------------------------------------------------------- 1
    rule("1. ACQUISITION — does teaching lower the loss on what was taught?")
    before = learner.sequence_loss(lesson_ex)
    canary_start = learner.canary_loss()
    print(f"  lesson: {LESSON[0]!r} -> {LESSON[1]!r}")
    print(f"  loss before teaching   {before:.4f}")
    for i in range(args.repeats):
        r = learner.observe(LESSON, weight=2.0, force=True)
        if i % 4 == 0 or i == args.repeats - 1:
            print(f"    pass {i+1:>2}  loss {r.loss_before:.4f} -> {r.loss_after:.4f}"
                  f"  drift {r.drift:.4f}")
    after_teaching = learner.sequence_loss(lesson_ex)
    drop = (before - after_teaching) / before * 100
    print(f"  loss after teaching    {after_teaching:.4f}   ({drop:+.1f}%)")
    acquisition_ok = after_teaching < before

    # ---------------------------------------------------------------- 2
    rule("2. RETENTION — does it survive learning many unrelated things?")
    n = 0
    for round_i in range(args.distractor_rounds):
        for turns in DISTRACTORS:
            learner.observe(turns)
            n += 1
    after_distractors = learner.sequence_loss(lesson_ex)
    regression = (after_distractors - after_teaching) / after_teaching * 100
    print(f"  learned {n} unrelated exchanges")
    print(f"  lesson loss now        {after_distractors:.4f}   "
          f"({regression:+.1f}% vs. straight after teaching)")
    print(f"  vs. never taught       {before:.4f}")
    retention_ok = after_distractors < before

    # ---------------------------------------------------------------- 3
    rule("3. STABILITY — is English itself still intact?")
    canary_end = learner.canary_loss()
    delta = (canary_end - canary_start) / canary_start * 100
    print(f"  held-out canary loss   {canary_start:.4f} -> {canary_end:.4f}  ({delta:+.1f}%)")
    print(f"  (these sentences are never trained on and never enter replay)")
    summary = learner.journal.summary()
    print(f"  updates applied {summary['updates_applied']}, "
          f"rollbacks {summary['rollbacks']}, "
          f"consolidations {summary['consolidations']}")
    stability_ok = canary_end < canary_start * 1.10

    # ---------------------------------------------------------------- 4
    rule("4. PERSISTENCE — does it survive a restart?")
    learner.save()
    del learner, model

    fresh_model, fresh_tok, fresh_cfg, _ = load_checkpoint(checkpoint, args.device)
    restored = resume_learned_weights(fresh_model, state_dir, args.device)
    # Note: no LoRA, no learner — a bare GPT loading a plain checkpoint.
    x, y = torch.tensor([lesson_ex[0]]), torch.tensor([lesson_ex[1]])
    fresh_model.eval()
    with torch.no_grad():
        _, reloaded_loss, _ = fresh_model(x, y)
    reloaded_loss = float(reloaded_loss)

    baseline_model, _, _, _ = load_checkpoint(checkpoint, args.device)
    baseline_model.eval()
    with torch.no_grad():
        _, baseline_loss, _ = baseline_model(x, y)
    baseline_loss = float(baseline_loss)

    print(f"  learned.pt restored    {restored}")
    print(f"  reloaded (plain GPT)   {reloaded_loss:.4f}")
    print(f"  untouched base model   {baseline_loss:.4f}")
    print(f"  the weights themselves differ by what was learned, with no")
    print(f"  adapters and no learner involved in the reload.")
    persistence_ok = reloaded_loss < baseline_loss * 0.98

    # ----------------------------------------------------------------
    rule("RESULT")
    checks = [
        ("acquisition", acquisition_ok, f"{before:.3f} -> {after_teaching:.3f}"),
        ("retention", retention_ok, f"{after_distractors:.3f} still below {before:.3f}"),
        ("stability", stability_ok, f"canary {delta:+.1f}%"),
        ("persistence", persistence_ok, f"{reloaded_loss:.3f} vs base {baseline_loss:.3f}"),
    ]
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<12} {detail}")

    if not args.keep_state:
        shutil.rmtree(state_dir, ignore_errors=True)
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
