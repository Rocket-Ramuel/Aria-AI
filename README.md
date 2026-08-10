# Aria

A small English language model, built from scratch, whose weights keep changing
while you talk to it.

Two halves:

1. **A real transformer language model**, trained from random initialisation on
   an English corpus. The tokenizer, the architecture, the training loop and the
   sampler are all implemented in this repository — nothing pretrained is
   downloaded.
2. **An online learning engine** that takes a gradient step on each
   conversational turn and writes the result back into the model's own weights,
   with the machinery required to make that survivable rather than destructive.

The second half is the interesting one. Doing continual learning *naively* —
one SGD step per turn on whatever the user just typed — reliably destroys a
language model within a few hundred turns. Most of this repo is about not doing
that.

---

## What you should expect from it

Read this before running it, because the honest framing matters.

A model you can pretrain on a laptop CPU in an evening is **six to seven orders
of magnitude smaller** than a frontier LLM, trained on about five orders of
magnitude less text. The `small` preset is ~7M parameters trained on ~4M tokens.
What that buys you:

- **Yes:** fluent-looking English word order, correct spelling, plausible
  syntax, topical continuity within a sentence or two, and *measurable,
  inspectable learning from conversation* — you can watch the loss on a
  specific exchange fall as you repeat and correct it, and confirm the change
  persists across restarts.
- **No:** factual reliability, reasoning, instruction-following, or coherence
  across a long reply. It will confabulate constantly. It is a demonstration of
  a learning mechanism, not a useful assistant.

If you want a model that is genuinely good at conversation *and* learns online,
the same `aria/learner.py` will attach to a much larger pretrained model — the
learning engine is independent of the model size. The bundled pretraining is
what makes this repo self-contained, not what makes it capable.

---

## Quick start

```bash
pip install -r requirements.txt

python -m aria prepare                      # fetch corpus, train BPE, build datasets
python -m aria pretrain --preset small \
    --steps 4000 --max-minutes 60           # train the base model
python -m aria chat --verbose               # talk to it; it learns as you do
```

`--verbose` prints the learner's decision after each turn, which is the point of
the whole exercise:

```
you> what is your name
aria> my name is Aria.
[learn] loss 3.412 -> 3.088  surprise +0.71  lr 2.4e-04  replay 6  drift 0.011
```

Everything the learner accumulates lives in `runs/aria/online/`:

| file | contents |
| --- | --- |
| `learned.pt` | the weights, with all learning merged in — a plain checkpoint |
| `replay.json` | remembered exchanges, as readable text |
| `journal.jsonl` | every decision: what was learned, skipped, or rolled back |
| `learner_state.json` | learning rate, surprise baseline, canary baseline |

Delete that directory to reset Aria to her post-pretraining state. The base
checkpoint is never modified.

---

## How the online learning works

Each turn is handed to `OnlineLearner.observe(turns)`, which runs six mechanisms.

### 1. Surprise gating — learn only from what is novel

The loss on the new exchange is computed *before* any update. If the model
already predicts it well (below a running EMA of recent losses), no gradient
step is taken at all; the exchange is only filed into memory. When a step is
taken, the learning rate scales with how surprising the exchange was.

This is both a compute saving and a regulariser: it stops the twentieth
"hello" from getting the same twenty gradient steps as the first.

### 2. Rehearsal — never train on a batch of only-new-data

Catastrophic forgetting is what happens when a network is shown a stream of
non-i.i.d. data. So the update batch is never only the new turn. It mixes:

- the new exchange,
- samples from a **persistent reservoir** of past conversations,
- samples from the **original pretraining corpus**.

The reservoir uses reservoir sampling, so old memories are never guaranteed to
be evicted just for being old, and corrections carry extra sampling weight
because "no, say it like this" is the highest-value signal available.

### 3. Elastic weight consolidation — protect the weights that matter

At the end of pretraining, `estimate_fisher` computes the diagonal empirical
Fisher information, `E[(∂L/∂θ)²]`, over the pretraining data. Large entries mark
weights that the model's existing knowledge is sensitive to.

Online updates add a penalty `λ · Σᵢ Fᵢ(θᵢ − θ*ᵢ)²`, pulling sensitive weights
back toward their anchor values while leaving insensitive ones free to move.
Knowing English is protected; the capacity to learn your name is not spent on
protecting it.

### 4. A trust region — bound the blast radius of one conversation

After every step each tensor is projected back so that
`‖θ − θ_anchor‖ / ‖θ_anchor‖ ≤ trust_radius` (default 5%). No single
conversation — including a deliberately adversarial one — can move the model
arbitrarily far. In LoRA mode the bound is applied to the effective weight
delta `BA·(α/r)` relative to the frozen base matrix.

### 5. A canary and automatic rollback — undo learning that made it worse

A fixed set of held-out English sentences is evaluated every `health_interval`
updates. These are never trained on and never enter the replay buffer. If loss
on them rises past tolerance for `health_patience` consecutive checks, the
learner restores its last known-good snapshot and halves its learning rate.

This is the mechanism that makes "always learning" not mean "always
degrading". Learning that damages the model is detected and reverted without
anyone watching.

### 6. Consolidation — make the learning permanent

Every `consolidate_interval` healthy updates, the current weights are accepted
as the new known-good state. In LoRA mode the adapters are **folded into the
base weight matrices** and reset to zero, so the knowledge moves into the
model's ordinary parameters and the trust region re-centres on the new
position. In `full`/`ffn` mode the EWC anchor moves toward the current weights
via an EMA.

This answers the "does it *really* adjust its own weights" question directly:
after consolidation, `learned.pt` is a plain checkpoint with no adapters in it,
loadable by a bare `GPT(cfg)`, and it differs from the base checkpoint exactly
by what it learned from you.

### What gets updated

`--learner-plasticity` chooses the surface that moves:

| mode | trainable | notes |
| --- | --- | --- |
| `lora` (default) | rank-8 adapters on `q/v/o/down` projections | base weights frozen between consolidations; safest, and consolidation still writes into the real weights |
| `ffn` | every feed-forward matrix | full-weight updates, EWC-protected |
| `full` | everything | maximum plasticity, highest risk; lower `trust_radius` if you use it |

---

## Commands

```bash
python -m aria prepare [--vocab-size 8192] [--block-size 256] [--offline]
python -m aria pretrain --preset {tiny,small,base} [--steps N] [--max-minutes M]
python -m aria chat [--verbose] [--no-learn] [--learner-plasticity full]
python -m aria sample --prompt "The " --state-dir runs/aria/online
python -m aria teach notes.txt          # learn from a document, paragraph by paragraph
python -m aria status                   # what the learner has been doing
```

Inside `chat`:

```
/status        learner and memory statistics
/memory [n]    recent remembered exchanges
/teach <text>  learn from a passage directly
/correct <text>  replace Aria's last reply with yours and learn from it (weight 3x, bypasses the gate)
/consolidate   force a consolidation pass
/learn on|off  toggle online learning
/save          write weights and memory to disk
```

`--no-learn` gives you a frozen model, which is the control condition when you
want to check that a change you are seeing is really coming from learning.

---

## The model

Standard decoder-only transformer, implemented in `aria/model.py`:

- pre-norm residual blocks with RMSNorm
- rotary position embeddings (RoPE)
- grouped-query attention (small KV cache during generation)
- SwiGLU feed-forward
- tied input/output embeddings
- byte-level BPE trained on the corpus itself, so any input is encodable

| preset | layers | d_model | heads | ctx | params |
| --- | --- | --- | --- | --- | --- |
| `tiny` | 4 | 128 | 4 | 128 | ~1M |
| `small` | 6 | 256 | 8 | 256 | ~7M |
| `base` | 12 | 768 | 12 | 512 | ~100M |

`tiny` exists so the whole pipeline runs end to end in minutes. `base` wants a
GPU.

### The chat format

```
<|bos|><|user|> hello there <|eot|><|aria|> hi <|eot|><|user|> ...
```

Loss is computed **only on tokens after an `<|aria|>` marker** — the model is
trained to produce replies, never to imitate the user. This matters more than
usual for online learning: without it, a model trained on live conversation
converges on parroting its input.

---

## Corpus

`prepare` fetches WikiText-2 and a public-domain prose collection (~19 MB of
modern English), cleans the markup artifacts, and builds three artifacts:
`tokenizer.json`, packed `train.bin`/`val.bin` token streams, and `chat.pt`.

`chat.pt` combines the hand-written conversation seed in
`data/seed_dialogues.txt` (which teaches the turn protocol and a baseline
register) with **surrogate dialogues** formed from adjacent corpus sentences.
The surrogate pairs are not real conversations and are labelled as such in the
code: they teach turn structure and topical continuity, nothing more. Real
conversational competence is what the online learner is for.

Bring your own corpus by dropping `.txt` files into `data/raw/` and running
`prepare --offline`.

---

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers the model (causal masking, KV-cache equivalence against a full
forward pass, LoRA merge fidelity, gradient flow), the tokenizer (lossless
round-trip on arbitrary bytes, special-token atomicity), and — most
importantly — each continual-learning safeguard individually: that the trust
region really caps drift, that rollback really restores weights, that
consolidation really moves knowledge into the base matrices, that rehearsal
really protects an old lesson from being erased by forty new ones, and that
the canary is never trained on.

---

## Limitations

- **Scale.** See the framing at the top. This model confabulates constantly.
- **Online learning is not memory.** A gradient step changes a *disposition*,
  not a retrievable fact. Telling Aria your name once makes that reply more
  likely; it does not create a lookup table. Facts you need reliably belong in
  a retrieval layer, which this repo does not implement.
- **Learning from a single user is a narrow distribution.** The safeguards bound
  the drift; they do not make it neutral. Over thousands of turns Aria will
  become specifically adapted to how *you* write.
- **Anything you type may end up in the weights and on disk** in
  `runs/aria/online/`. Use `--no-learn` for anything you would not want stored.
- **CPU only by default.** `--device cuda` works, but the presets are sized for
  a CPU budget.

## Layout

```
aria/
  config.py      dataclass configs and size presets
  tokenizer.py   byte-level BPE, trained from scratch
  model.py       the transformer, KV cache, LoRA attach/merge
  data.py        corpus prep, chat formatting, batching
  pretrain.py    offline training loop + Fisher estimation
  learner.py     the online learning engine
  memory.py      replay buffer and decision journal
  sample.py      generation
  chat.py        REPL
  cli.py         command line
tests/           53 tests
data/seed_dialogues.txt   hand-written conversation seed
```
