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

You can also hand it **documents** — a writing sample, a chat log, the
transcript of someone talking — and it will learn to sound like them, and, for
a transcript, to answer the way a chosen person answers. And if you want
nothing between it and the people it learns from, you can start it **blank**:
no pretraining, no vocabulary, no grammar, only what you give it.

The online learning is the interesting part. Doing continual learning *naively* —
one SGD step per turn on whatever the user just typed — reliably destroys a
language model within a few hundred turns. Most of this repo is about not doing
that.

---

## What you should expect from it

Read this before running it. A model you can pretrain on a laptop CPU is **six
to seven orders of magnitude smaller** than a frontier LLM, trained on five
orders of magnitude less text. The shipped model is 6.5M parameters trained on
4.85M tokens for 8000 steps, reaching a validation loss of 3.95 (perplexity
~52). Here is what that actually looks like, unedited.

**Free-running text** — grammatical, correctly spelled, structurally coherent:

> **The river** and the western limbs of aneurysm. If this is not a result of the colour, it is no longer the chief form of infection.

**Conversation on ground the seed dialogues cover** — fluent and on-topic,
but this is largely recall: these exchanges are in
[`aria/seed_dialogues.txt`](aria/seed_dialogues.txt), and the model has
memorised them.

> **you>** what is your name?
> **aria>** My name is Aria.
>
> **you>** can you learn?
> **aria>** Yes. Every time we talk I adjust my own weights a little, so I change over time.

**Conversation off that ground** — grammatical English, no meaning:

> **you>** what did you have for breakfast?
> **aria>** It is a well-known, strong and often.
>
> **you>** I just finished building a bookshelf
> **aria>** There were also seven times in the year.

So, plainly:

- **Yes:** real English syntax and spelling, a conversational register, and
  *measurable, inspectable learning from conversation* — you can watch the loss
  on a specific exchange fall as you teach it, verify that teaching forty new
  things does not erase it, and confirm it survives a restart. Those numbers are
  [below](#does-the-learning-actually-work), and you can reproduce them.
- **No:** factual reliability, reasoning, instruction-following, or coherence
  beyond a sentence. It confabulates constantly. It is a working demonstration
  of a learning mechanism, not a useful assistant.

The learning engine in `aria/learner.py` does not care how big the model is —
it will attach to a much larger pretrained one and behave the same way. The
bundled pretraining is what makes this repo self-contained, not what makes it
capable.

---

## Running it

It runs on your own machine, on the CPU, for free. No API key, no account, no
service to sign up for, and nothing is sent anywhere — the only network access
in the whole project is `prepare` downloading the public-domain corpus once.

**Requirements:** Python 3.10+ and about 2 GB of RAM. Any laptop from the last
decade will do. macOS, Linux and Windows all work; a GPU is optional.

### Talk to her straight away

A trained model ships in the repo, so there is nothing to train before you can
use it:

```bash
git clone https://github.com/Rocket-Ramuel/Aria-AI.git
cd Aria-AI
pip install -e .

aria serve      # browser UI, or `aria chat --verbose` for the terminal
```

`checkpoints/aria-small.pt` (30 MB) is the 6.5M-parameter model described
above, stored in half precision and loaded back into a float32 model. It
carries its own tokenizer and Fisher information, so nothing else is needed.

### Train your own

```bash
aria quickstart
```

`quickstart` builds the corpus, trains the model for 45 minutes (adjust with
`--minutes`), and opens a chat page in your browser. Training is **resumable** —
run it again and it picks up where it stopped, so you can train in short
sittings. A model you train yourself is preferred over the shipped one
automatically.

In a hurry? `aria quickstart --preset tiny --minutes 5` gives you something to
talk to almost immediately. It will not be good, but the learning machinery is
identical and you can watch it work.

### Or step by step

```bash
aria prepare                 # fetch corpus, train the BPE vocabulary, pack datasets
aria pretrain --preset small --steps 5000 --max-minutes 60
aria serve                   # browser UI at http://localhost:8000
aria chat --verbose          # or stay in the terminal
```

Everything also works as `python -m aria <command>` without installing.

### The browser UI

`aria serve` starts a local page on `127.0.0.1:8000`. It is one inline HTML
file served by Python's standard library — no framework, no build step, no CDN,
so it works with the network cable unplugged. Replies stream token by token,
and each turn shows what the learner decided.

It binds to localhost. Anything typed into that page gets written into the
model's weights and to disk, so don't put it on a public interface.

Binding to localhost does not on its own stop *other web pages* in your browser
from talking to it, so the server also refuses any request whose `Host` or
`Origin` isn't its own page, and any POST that isn't JSON. That blocks
cross-site form posts and DNS-rebinding pages. If you deliberately serve it on
your network, name the hosts it may be reached by with `--allow-host`.

### The terminal

```
you> what is your name
aria> my name is Aria.
[learn] loss 3.412 -> 3.088  surprise +0.71  lr 2.4e-04  replay 6  drift 0.011
```

That diagnostic line is the point of the whole exercise. `--verbose` turns it on.

### Free GPU, if you want a bigger model

[`notebooks/Aria_on_Colab.ipynb`](notebooks/Aria_on_Colab.ipynb) runs the whole
pipeline on Google Colab's free tier. A T4 trains the `small` preset in a few
minutes and makes the ~100M-parameter `base` preset realistic. The last cell
packages the trained weights so you can download them and carry on locally —
talking to Aria needs no GPU, only training benefits from one.

### How slow is CPU, really?

On four CPU cores, the `small` preset runs about 70 training steps per minute,
and a chat reply takes a second or two. Each turn's learning update costs about
as much as one more reply. It is comfortably interactive.

Everything the learner accumulates lives in an `online/` directory next to the
checkpoint it is learning on top of — `checkpoints/online/` for the shipped
model, `runs/aria/online/` for one you trained, `runs/blank/online/` for a
blank one. (`.gitignore` excludes every `online/` directory: the replay buffer
and journal contain your conversations in plain text.)

| file | contents |
| --- | --- |
| `learned.pt` | the weights, with all learning merged in — a plain checkpoint |
| `replay.json` | remembered exchanges, as readable text |
| `journal.jsonl` | every decision: what was learned, skipped, or rolled back |
| `learner_state.json` | learning rate, surprise baseline, canary baseline |

Delete that directory to reset Aria to her post-pretraining state. The base
checkpoint is never modified. `aria status` reads the same directory `chat`
and `serve` write to.

---

## Teaching it a voice

### Upload a document

In the browser, open **Teach from a file** under the message box (or drop a
file anywhere on the page). In the terminal, `/upload path/to/file`. In bulk,
`aria teach file1.txt file2.docx ...`.

Readable formats: `.txt`, `.md`, `.docx`, `.srt`/`.vtt` subtitles (timings
and cue numbers are stripped, leaving what was said), and `.pdf` if you
`pip install pypdf`. Audio is not supported — turning speech into text needs a
speech-recognition model, which this project doesn't include. Transcribe the
recording first and upload the transcript.

What happens to the text depends on what it is:

- **Prose** — an essay, letters, a diary — is cut into context-sized windows
  and learned *as if Aria had written it*. It is also cut into a run of
  exchanges, each sentence answering the one before, so the voice is learned as
  a way of *replying*, not only of continuing text.
- **A transcript or chat log** — lines like `Sam: are you coming tonight?` —
  becomes conversations when you name a speaker (`as Jo` in the terminal, the
  name box in the browser, `--speaker Jo` for `teach`). Jo's lines become
  Aria's side and everyone else's become the prompts, so she learns how Jo
  *answers*. Without a name, the labels are stripped and it is learned as prose.

An upload takes several passes over the material (default 3, at least 24 steps,
at most 400) at three times the chat learning rate, with rehearsal mixed in and
every safeguard below still running. On the shipped model, a 200-word sample
drops from loss 6.4 to 3.8 in about 6 seconds with held-out English unchanged.
A sample of each upload is kept in the replay buffer, so later conversations
keep rehearsing it.

### Your own messages

By default Aria also learns from what *you* type, framed as something she
said (`--learner-style-mirror false` turns it off). Over a conversation that
pulls her phrasing toward yours. Her own replies are learned too, at the same
time; the blank model below skips those, since rehearsing her own babble would
teach her nothing.

### Start blank: no pretraining at all

```bash
aria serve --blank        # or: aria chat --blank, aria teach --blank sample.txt
```

`--blank` uses `runs/blank/base.pt`, creating it if needed (`aria blank` makes
one explicitly, `--size tiny|small|base`). It is a model with random weights
and a byte-level tokenizer with no learned vocabulary: it assumes nothing
about English, spelling or grammar. Everything it ever produces it learned
from what you uploaded and said, so its voice can only be the voice of its
sources.

It runs with different learner settings, stored in its checkpoint: full
plasticity, a 10× higher learning rate, and none of the anchors, trust region,
EWC or canary. Those exist to protect knowledge a model already has, and a
blank model has none to protect.

**Expect it to be slow to talk.** Measured on a CPU, uploading a 1,700-word
writing sample:

| after | loss | a reply to "what do you like?" |
| --- | --- | --- |
| 32 steps (~20 s) | 2.7 | `endrog s weios,at,lllr. noanthate a maudiorep.` |
| 190 steps (~2 min) | 1.0 | `I have on pay.` |
| 580 steps (~6 min) | 0.14 | `Thart made cors on before conversations, I do not have a uext I will day ow.` |

It finds letters, then words, then the shape of a sentence in the source's
style — but 1,700 words is far too little to learn a language from, and it
mostly recombines what it read. Give it tens of thousands of words of one
person's writing or transcribed speech and run several uploads; it will stay a
mimic, not a conversationalist. If you want sensible replies *in* someone's
voice, the pretrained model plus uploads gets you there much sooner.

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

If you want a step on *literally* every turn, pass
`--learner-surprise-gate false`. Corrections made with `/correct` bypass the
gate regardless.

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
Fisher information, `E[(∂L/∂θ)²]`, over the pretraining data, with one
backward pass per sequence (squaring a batch-averaged gradient would measure
something smaller). Large entries mark
weights that the model's existing knowledge is sensitive to.

Online updates add a penalty `λ · Σᵢ Fᵢ(θᵢ − θ*ᵢ)²`, pulling sensitive weights
back toward their anchor values while leaving insensitive ones free to move.
Knowing English is protected; the capacity to learn your name is not spent on
protecting it.

### 4. A trust region — bound the blast radius of one conversation

After every step each tensor is projected back so that
`‖θ − θ_anchor‖ / ‖θ_anchor‖ ≤ trust_radius` (default 5%). In LoRA mode the
bound is applied to the effective weight delta `BA·(α/r)` relative to the
frozen base matrix.

The anchor moves at each consolidation, so this bounds how far the model can
go *between* consolidations — every 32 updates by default — not over its whole
life. A long conversation, or a big upload, can carry it further one bounded
stretch at a time; what bounds the total is the canary below, which refuses to
consolidate, and rolls back, once general English starts to suffer.

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

## Does the learning actually work?

`scripts/demo_learning.py` is an experiment, not a demo. It teaches the model
one novel fact, then tries to break it, and prints the numbers it was judged
on. Run it yourself:

```bash
python scripts/demo_learning.py
```

Measured on the shipped checkpoint, teaching
`"who wrote the notes in the blue folder"` → `"Priya wrote the notes in the
blue folder last Tuesday"` over 12 passes, then learning 40 unrelated
exchanges:

| check | what it asks | result |
| --- | --- | --- |
| **acquisition** | does teaching lower the loss on what was taught? | 6.68 → **3.14** (53% lower) |
| **retention** | does the lesson survive 40 unrelated new ones? | **1.79** — still far below the untaught 6.68 |
| **stability** | is held-out English intact afterwards? | canary loss **−0.1%**, 0 rollbacks |
| **persistence** | does it survive a restart? | a bare `GPT` loading a plain checkpoint gets **1.79** vs base **6.68** |

The stability row is the one that matters. Fifty-two gradient updates went into
the model during that run and its English did not degrade — that is the whole
point of the six mechanisms below. The persistence row is the answer to "does
it *really* change its own weights": after consolidation there are no adapters
left in `learned.pt`, only ordinary weight matrices that differ from the base
model by what it learned.

Trust-region drift sat at exactly its 0.05 cap throughout, which is the bound
doing its job rather than a coincidence.

---

## Commands

```bash
aria quickstart [--preset tiny] [--minutes 45]   # everything, in one go
aria prepare [--vocab-size 8192] [--block-size 256] [--offline]
aria pretrain --preset {tiny,small,base} [--steps N] [--max-minutes M]
aria serve [--port 8000] [--no-browser]          # browser UI
aria chat [--verbose] [--no-learn] [--learner-plasticity full]
aria sample --prompt "The " --state-dir runs/aria/online
aria serve --blank            # a model with no pretraining (see above)
aria blank [--size small]     # create one explicitly
aria teach notes.txt more.docx         # learn from documents
aria teach chat.txt --speaker Jo       # learn to answer like Jo
aria status                   # what the learner has been doing
```

Inside `chat`:

```
/status        learner and memory statistics
/memory [n]    recent remembered exchanges
/teach <text>  learn from a passage directly
/upload <file> [as <name>]   learn from a whole document or transcript
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
`aria/seed_dialogues.txt` with **surrogate dialogues** formed from adjacent
corpus sentences. The surrogate pairs are not real conversations and are
labelled as such in the code: they teach turn structure and topical continuity,
nothing more.

The seed is ~150 short exchanges, repeated (`--seed-repeat`, default 100) to
about a quarter of the chat examples. At this model size that means the seed is
substantially **memorised** rather than generalised — which is exactly what the
transcripts above show, and why the conversational surface is thin the moment
you step off it. It is a starting register, not knowledge. Retune the mix
without re-running BPE:

```bash
aria prepare --chat-only --seed-repeat 200
```

Blocks with near-zero sentence-punctuation density are dropped before training:
public-domain dumps tend to carry word lists, indices and code appendices that
teach a small model nothing. The train/validation split holds out sixteen
evenly spaced chunks rather than one tail slice, so validation loss reflects
the whole corpus instead of whichever source happens to be last.

Bring your own corpus by dropping `.txt` files into `data/raw/` and running
`prepare --offline`. No network access is needed in that mode.

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
the canary is never trained on. Uploads have their own tests (every format,
transcript parsing, that a transcript trains only the chosen voice, that a
blank model learns from a sample), and so does the server's refusal of
requests from other sites.

---

## Limitations

- **Scale.** See the framing at the top. This model confabulates constantly.
- **Online learning is not memory.** A gradient step changes a *disposition*,
  not a retrievable fact. Telling Aria your name once makes that reply more
  likely; it does not create a lookup table. Facts you need reliably belong in
  a retrieval layer, which this repo does not implement.
- **The conversational register is memorised, not learned.** The base model
  recalls the seed dialogues closely and has little to say beyond them. Online
  learning changes dispositions on top of that; it does not substitute for a
  bigger model or a real dialogue corpus.
- **Learning from a single user is a narrow distribution.** The safeguards bound
  the drift; they do not make it neutral. Over thousands of turns Aria will
  become specifically adapted to how *you* write.
- **Anything you type or upload may end up in the weights and on disk** in the
  checkpoint's `online/` directory. Use `--no-learn` for anything you would not
  want stored. A model that has learned someone's voice can reproduce their
  writing, sometimes verbatim — treat `learned.pt` as you would the documents
  you taught it.
- **A blank model is a mimic.** Starting from nothing, it needs a great deal of
  text before it says anything coherent, and what it says recombines its
  sources. See [Start blank](#start-blank-no-pretraining-at-all).
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
  documents.py   reading uploads: txt/md/docx/srt/vtt/pdf, transcripts
  sample.py      generation
  chat.py        REPL
  serve.py       local browser UI (standard library only)
  cli.py         command line
  seed_dialogues.txt        hand-written conversation seed
tests/           151 tests
notebooks/       Colab notebook for a free GPU
scripts/demo_learning.py  measures whether the learning actually works
```
