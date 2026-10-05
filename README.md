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

You can also give it **documents**, right in the chat: press the paperclip,
pick a book, a pile of letters, a chat log or a speech transcript — any size —
and Aria reads it and learns its grammar, words and voice in the background
while you keep talking to her. For a transcript she can learn to answer the way
one chosen person answers. And if you want nothing between her and the people
she learns from, you can start her **blank**: no pretraining, no vocabulary, no
grammar, only what you give her.

However much she learns, she doesn't get bigger: learning changes the values of
her weights, never their number. The shipped model is a 20 MB file and stays
that size. [Details below](#how-big-does-aria-get).

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

**Requirements:** Python 3.10+ and about 1.5 GB of free RAM (measured peak:
0.85 GB for the pretrained model, 1.05 GB for a blank one, most of it PyTorch
itself). Any laptop from the last
decade will do. macOS, Linux and Windows all work; a GPU is optional and
[used automatically](#gpus-nvidia-and-apple-silicon) if you have one.

### Talk to her straight away

A trained model ships in the repo, so there is nothing to train before you can
use it:

```bash
git clone https://github.com/Rocket-Ramuel/Aria-AI.git
cd Aria-AI
pip install -e .

aria serve      # browser UI, or `aria chat --verbose` for the terminal
```

`checkpoints/aria-small.pt` (20 MB) is the 6.5M-parameter model described
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

The **paperclip** beside the message box (or dropping a file anywhere on the
page) gives her a document to read; see [below](#teaching-it-from-documents).
The menu at the top switches between the **pretrained** model and a **blank**
one; each keeps its own memory, and the blank one is created the first time you
pick it.

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

### GPUs: NVIDIA and Apple Silicon

Every command takes `--device`, and the default, `auto`, picks for you:

1. an **NVIDIA GPU** (`cuda`), if PyTorch was installed with CUDA support;
2. the **GPU of an Apple Silicon Mac** (`mps`) — any M1, M2, M3 or M4 Mac,
   including a MacBook Air;
3. otherwise the **CPU**.

`aria serve` prints which it chose ("running on the Apple GPU"). Before a GPU
is trusted, Aria runs a short self-test on it — a tiny model doing everything
she does: learning, the low-memory optimiser, cortical areas, sampling and
memory recall. If anything fails (an operation your PyTorch version doesn't
support on that GPU, say), she says so and runs on the CPU instead of
crashing mid-conversation. On a Mac, PyTorch is also told to run any
operation the Apple GPU lacks on the CPU instead of failing.

**On a Mac:** install PyTorch normally (`pip install torch`; the Mac build
includes Apple GPU support) and run `aria serve` as usual. Intel Macs have no
supported GPU and use the CPU. Two things to know:

- For chatting, the GPU isn't necessarily faster: Aria is small enough that
  handing each word to the GPU and back can cost as much as it saves. It helps
  most with big uploads. To compare on your machine, try `--device cpu` and
  `--device mps` on the same upload.
- A MacBook Air has no fan. Under a long upload — on the GPU or the CPU — it
  slows itself down to stay cool, which is normal and harmless but caps how
  fast big jobs go. Plug in for big uploads; they use the battery heavily.

**Tested:** the device selection, the self-test and the fallback are covered
by the test suite (with GPUs simulated), and everything was measured on a CPU.
Aria has **not yet been run on a real Apple GPU**. The parts known to differ
on one were made safe (weights load through the CPU, the optimiser's momentum
stays 32-bit there, 16-bit arithmetic is off there), and the self-test is the
backstop. If something goes wrong on your Mac, the message says what.

### How slow is CPU, really?

On four CPU cores, the `small` preset runs about 70 training steps per minute,
and a chat reply takes a second or two. Each turn's learning update costs about
as much as one more reply. It is comfortably interactive. Reading documents
is slower — see [how long](#how-long-it-takes).

Everything the learner accumulates lives in an `online/` directory next to the
checkpoint it is learning on top of — `checkpoints/online/` for the shipped
model, `runs/aria/online/` for one you trained, `runs/blank/online/` for a
blank one. (`.gitignore` excludes every `online/` directory: the replay buffer
and journal contain your conversations in plain text.)

| file | contents |
| --- | --- |
| `learned.pt` | the weights, with all learning merged in — a plain checkpoint, half precision |
| `replay.json` | remembered exchanges, as readable text (at most 4,096) |
| `journal.jsonl` | every decision: what was learned, skipped, or rolled back (rotates at 4 MB) |
| `learner_state.json` | learning rate, surprise baseline, canary baseline |

Delete that directory to reset Aria to her post-pretraining state. The base
checkpoint is never modified. `aria status` reads the same directory `chat`
and `serve` write to.

---

## Teaching it from documents

### Upload a document, then keep talking

In the browser, press the **paperclip** next to the message box, or drop files
anywhere on the page. A card appears in the conversation showing her progress;
you can go on chatting while she reads, and a message waits for at most one
learning step (a fraction of a second), not for the document. **Stop** on the
card ends it early and keeps everything learned so far.

In the terminal, `/upload path/to/file` — or just drag the file into the
terminal window, which pastes its path. Ctrl-C stops early and keeps what she
learned. For many files at once, `aria teach a.txt b.docx c.srt`.

Readable formats: `.txt`, `.md`, `.docx`, `.srt`/`.vtt` subtitles (timings
and cue numbers are stripped, leaving what was said), and `.pdf` if you
`pip install pypdf`. Audio is not supported — turning speech into text needs a
speech-recognition model, which this project doesn't include. Transcribe the
recording first and upload the transcript.

### What she learns from it

- **Prose** — a book, letters, a diary — is learned *as if Aria had written
  it*: cut into context-sized windows, which teach the grammar, vocabulary and
  rhythm of continuous text; and into a run of exchanges, each sentence
  answering the one before, so the same language is learned as a way of
  *replying*. For a model with no pretraining, the second is the difference
  between memorising a sample and answering in its style.
- **A transcript or chat log** — lines like `Sam: are you coming tonight?` —
  is noticed before it is sent, and the page asks whose way of talking to
  learn (the terminal asks too; `teach` takes `--speaker Jo`). Jo's lines
  become Aria's side and everyone else's become the prompts, so she learns how
  Jo *answers*. Choose nobody and the labels are stripped and it is learned as
  prose.

### No limits on size

There is no cap on the size of a document or on how long she learns from it.
Every pass covers every page. Files are streamed from disk and never held in
memory whole: reading a 30 MB document used no more memory than a 1 MB one
(measured, [below](#how-big-does-aria-get)). Long uploads are saved every five
minutes, so closing the laptop costs minutes, not hours.

Each upload takes three passes by default (four for a blank model), and at
least 24 steps for very short ones, at three times the chat learning rate,
with rehearsal of earlier lessons mixed in and every safeguard below still
running. The canary may roll back a stretch that hurt her general English; the
upload then carries on more gently, and stops only if that happens three times.

### Is she learning the language, or memorising?

One sentence-group in every twenty is held back and never trained on. The
progress card reports loss on that **held-out** text: if it falls, she is
learning the language of the document — grammar, word order, vocabulary — and
not just remembering its sentences. (Documents shorter than about 160
sentences are too small to hold any back; their card says plain "loss".)

### How long it takes

Measured on four CPU cores:

| model | reads, per pass | a 100,000-word book |
| --- | --- | --- |
| pretrained | ~770 words/s | ~7 min (3 passes) |
| blank | ~450 words/s | ~15 min (4 passes) |

A 5-million-word archive is an overnight job on a laptop. It runs in the
background and can be stopped any time; a GPU ([see below](#gpus-nvidia-and-apple-silicon)) is much
faster.

### Your own messages

By default Aria also learns from what *you* type, framed as something she
said (`--learner-style-mirror false` turns it off). Over a conversation that
pulls her phrasing toward yours. Her own replies are learned too, at the same
time; the blank model skips those, since rehearsing her own babble would teach
her nothing.

### Start blank: no pretraining at all

Pick **Blank** in the menu at the top of the page, or:

```bash
aria serve --blank        # or: aria chat --blank, aria teach --blank sample.txt
```

That uses `runs/blank/base.pt`, creating it if needed (`aria blank` makes one
explicitly, `--size tiny|small|base`). It is a model with random weights and a
byte-level tokenizer with no learned vocabulary: it assumes nothing about
English, spelling or grammar. Everything it ever produces it learned from what
you uploaded and said, so its grammar and its voice can only be those of its
sources.

It runs with different learner settings, stored in its checkpoint: full
plasticity, a 10× higher learning rate, and none of the anchors, trust region,
EWC or canary. Those exist to protect knowledge a model already has, and a
blank model has none to protect. Skipping them also saves two full copies of
the network in memory.

**Expect it to be slow to talk.** Measured on a CPU, uploading the same
1,700-word writing sample repeatedly:

| after | held-out loss, per byte | replies to "what do you like?" |
| --- | --- | --- |
| 31 steps (~6 s) | 2.61 (from 5.61) | `nd t ws w.` |
| 186 steps (~1.5 min) | 1.38 | `I word foremeter wonversiate, I spetteding ow.` |
| 1,116 steps (~10 min) | 1.13 | `What would you like me to say not?` · `Wand I will rme.` · `Yod corrections.` |

The loss is measured on sentences of the sample she never trained on, so its
fall is the language being learned, not the sample being memorised.

It finds letters, then words, then the shape of a sentence in the source's
style — but 1,700 words is far too little to learn a language from, and it
mostly recombines what it read. Grammar needs volume: give it hundreds of
thousands of words — a few books' worth of one person's writing or transcribed
speech — and it has something to generalise from. If you want sensible replies
*in* someone's voice soon, the pretrained model plus uploads gets you there
much faster.

---

## Aria's brain

Aria is organised loosely like a brain: separate parts with separate jobs.
None of this is a simulation of neurons; each part is an engineering
mechanism chosen because it does the job its namesake does. `/brain` shows
the map and what each part is doing.

| part | in a brain | in Aria |
| --- | --- | --- |
| **cortex** | slow learning of general knowledge over many repetitions | the transformer's weights, changed a little by each exchange |
| **hippocampus** | remembers an experience after one exposure; recalls it when something similar comes up | a fast memory of recent moments, used when she speaks (`aria/hippocampus.py`) |
| **sleep** | replays the day's memories and makes them permanent | consolidation: learning folded into the weights, memories re-encoded |
| **novelty** | dopamine marks surprising things as worth learning | the surprise gate: familiar exchanges are skipped, surprising ones learned harder |
| **cortical areas** | regions specialise in different work | optional: each feed-forward layer split into specialist areas (below) |
| **growth** | — | new layers added when she runs out of room |

### The hippocampus: one-shot memory

Her cortex learns slowly: one gradient step on "my dog is called Biscuit"
barely changes anything. The hippocampus stores every exchange she hears as
the cortex's internal state at each word, paired with the word that came
next. When she speaks, it first recalls the *episodes* the conversation is
about — matching its rarer words, as the brain uses context to cue a memory —
and then, word by word, blends in what came next at the closest remembered
moment. During sleep every memory is re-encoded, so memories keep matching as
the cortex changes. It holds 16,384 words (`--learner-hippocampus-tokens`, 0
to turn it off), about **8 MB**, and makes replies ~25% slower (46 → 58 ms).
With `--verbose` (or in the page) you see what she recalled:
`[recall] remembered "my dog is called Biscuit / ..." (match 0.73)`.

Measured on the shipped model, told five facts once each, then asked about
them. The number is the rank of the right word among her 8,192 possible next
words, where the answer needs it:

| | Biscuit | turquoise | carpenter | Priya | Wexford |
| --- | --- | --- | --- | --- | --- |
| no hippocampus | 147 | 5,524 | 543 | 3,926 | 2,405 |
| one learning step per fact (the cortex alone) | 137 | 5,590 | 457 | 3,113 | 2,244 |
| hippocampus | **1** | **2** | **1** | 29 | 2,216 |
| hippocampus, among 1,000 unrelated memories | **1** | **2** | 546 | 1,539 | 2,406 |

Loss on unrelated English was unchanged or slightly better throughout
(4.62 → 4.51–4.66).

**The honest limit:** the small model can't *use* a fact to answer a
question, even with the fact sitting in its context window — asked "what is
my dog called?" right after being told, it said "Biscuit" in 0 of 10 tries,
with or without memory. The hippocampus makes the right word her top choice
once she is answering in the right frame ("your dog is called …"), but the
6.5M-parameter cortex rarely gets there by itself. Recall pays off as the
cortex gets more capable — by growing, or with a larger model — and the
mechanism doesn't need to change for that.

### Cortical areas: specialists

`aria blank --areas 4` builds a model whose feed-forward layers are split into
four specialist areas, with a router (the thalamus's job) sending each word to
the two it scores highest. Nobody tells the areas what to specialise in;
`/brain` shows how unevenly the work ends up shared (in one run, one area of
layer 4 took 53% of the words, another 14%).

Measured learning the same 400 KB of text from blank for 400 steps:

| | size | time | held-out loss |
| --- | --- | --- | --- |
| dense (the default) | 4.49M | 126 s | 1.795 |
| 4 areas, same size (`--area-scale 1`) | 4.50M | 127 s | 1.900 (6% worse) |
| 4 areas, twice as wide (`--area-scale 2`) | 7.74M (+72%) | 156 s (+23%) | 1.774 (1% better) |

So at this scale areas make her more unusual, not smarter: the same-size
version learns a little worse, and the better one costs 72% more size for 1%.
That is why they are off by default. Mixture-of-experts layers like this pay
off in much larger models trained on much more text.

---

## How big does Aria get?

**The same size, until she grows.** A neural network's size is fixed by its
shape — how many layers, how wide — not by how much it has learned. Learning
changes the values of the weights; it never adds any. An Aria that has read a
hundred books is exactly as large as a fresh one. The only thing that makes her
bigger is [growing](#growing-more-room-to-learn), which you control.

Measured:

| | on disk | peak RAM |
| --- | --- | --- |
| shipped model (`checkpoints/aria-small.pt`) | 19.7 MB | — |
| learned weights, pretrained (`learned.pt`) | 13.1 MB, after any amount of learning | 0.85 GB |
| learned weights, blank (`learned.pt`) | 9.0 MB, after any amount of learning | 1.05 GB |
| reading a 1 MB document | — | same as above |
| reading a 30 MB document | — | same as above |

About 0.5 GB of the RAM is PyTorch itself, and 0.15 GB more is PyTorch's
optimiser machinery, loaded once. Aria's own share is around 0.1 GB.

What keeps it small (`aria/storage.py`):

- weights are stored in **half precision** and loaded back into a full-precision
  model — half the size, with outputs that differ by less than sampling noise;
- the input embedding and output layer are **one shared matrix**, stored once;
- the Fisher information takes **one byte per weight** on a log scale. (It used
  to be float16, which rounded 11% of the shipped model's values to zero;
  `scripts/recompute_fisher.py` re-estimated it, and the shipped file shrank
  from 30 MB to 20 MB with identical outputs);
- every save is **atomic**, so a crash mid-save never corrupts what was learned;
- nothing else grows without bound: the replay buffer keeps at most 4,096
  exchanges, the journal rotates at 4 MB while keeping lifetime totals, and an
  uploaded file is deleted once learned.

`aria status` and the page header show how much disk her memory uses. To carry
everything she has learned as one file: `aria export --with-learning --out
my-aria.pt` (13 MB for the small model). Add `--int8` for one byte per weight
— the shipped model becomes 13.2 MB instead of 19.7, and its loss on held-out
WikiText moves from 3.6896 to 3.6901 (+0.02%). That's for sharing and
archiving; the weights she is learning in stay 16-bit, because rounding to 8
bits on every save would erase small lessons.

### Growing: more room to learn

The other side of a fixed size is a fixed capacity. A 6.5M-parameter network
can only hold so much; past some point new learning overwrites old, which
rehearsal slows but cannot stop. So she can **grow**: `/grow` (in the page or
the terminal) adds a layer on top, `/grow 3` adds three.

A new layer starts out doing exactly nothing — its output projections are
zero — so the grown model gives *bit-for-bit* the same answers and loses
nothing she learned. Gradients still reach it, and it starts contributing as
soon as she learns again. In the default mode, the original layers keep
learning through their protected adapters; grown layers have no prior
knowledge to protect and are trained directly. Growth is saved with her
learned weights and restored in every later session.

She also **grows by herself** when she has read more than her size has room
for. The rule of thumb from scaling-law research is about 20 tokens of
training text per parameter; past that, a bigger model learns more from the
same text than more passes through a small one. The page header shows how
much room is left (the shipped model's pretraining used a quarter of it).
When it reaches zero, the next upload ends with one more layer, up to twice
her original depth (`--learner-grow-max-factor`, 0 to turn automatic growth
off). Each layer of the small model costs 1.5 MB on disk.

### A larger, smarter model, kept manageable

For a much larger model — `aria blank --size base` is 76M parameters as a
blank model — memory saving switches on by itself (above 20M parameters,
`--learner-memory-saver on|off|auto`):

- a **low-memory optimiser** (`aria/optim.py`): 2 bytes of bookkeeping per
  weight instead of AdamW's 8 — momentum in bfloat16, and the second moment of
  each matrix kept as one value per row and per column (as in Adafactor);
- **activation checkpointing**: during learning only each layer's input is
  kept, and the rest is recomputed when needed;
- **16-bit arithmetic** where the CPU (or GPU) has native bfloat16 — weights
  stay 32-bit, only the matrix maths runs in 16 bits;
- **half-precision safety snapshots**, and gradients freed straight after
  each step;
- weight files are **memory-mapped** on load instead of read into RAM first.

Measured on the 76M-parameter blank model, learning from a document on four
CPU cores:

| | before | after |
| --- | --- | --- |
| peak RAM while learning | 3,385 MB | **1,926 MB** (−43%) |
| RAM after loading | 1,161 MB | **866 MB** |
| time per learning step | 1.96 s | **1.45 s** |
| chatting only | — | 1,040 MB; 240-token replies in ~0.5 s |
| learned weights on disk | 151 MB | 151 MB (76 MB as an `--int8` export) |

The 16-bit arithmetic is what made it faster: 2.4× per step on this CPU,
which has native bfloat16. On a CPU without it, Aria detects that and stays
32-bit, and learning is about a third slower than before, because of the
recomputation. Small models are unchanged: below 20M parameters none of this
is needed.

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
on. Run it yourself (it uses the shipped model, or one you trained):

```bash
python scripts/demo_learning.py
```

Measured on the shipped checkpoint, teaching
`"who wrote the notes in the blue folder"` → `"Priya wrote the notes in the
blue folder last Tuesday"` over 12 passes, then learning 40 unrelated
exchanges:

| check | what it asks | result |
| --- | --- | --- |
| **acquisition** | does teaching lower the loss on what was taught? | 6.68 → **2.5–2.6** (~62% lower) |
| **retention** | does the lesson survive 40 unrelated new ones? | **0.82–1.02** — far below the untaught 6.68 |
| **stability** | is held-out English intact afterwards? | canary loss **+0.7% to +2.7%** across runs (tolerance 6%), 0 rollbacks |
| **persistence** | does it survive a restart? | a bare `GPT` loading the half-precision `learned.pt` gets **0.82**, the same as before saving, vs base **6.68** |

The stability row is the one that matters. Fifty-two gradient updates went into
the model during that run and its English barely moved — that is the whole
point of the six mechanisms below. (The lesson's loss keeps falling during
the 40 unrelated updates because it sits in the replay buffer and is
rehearsed alongside them — rehearsal doing its job.) The persistence row is the answer to "does
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
aria teach notes.txt more.docx         # learn from documents, any size
aria teach chat.txt --speaker Jo       # learn to answer like Jo
aria status [--blank]         # what the learner has been doing
aria blank --areas 4          # a blank brain with specialist cortical areas
aria export --with-learning --out my-aria.pt   # everything she knows, one file
aria export --with-learning --int8 --out my-aria.pt   # the same at one byte per weight
```

Inside `chat`:

```
/status        learner and memory statistics
/memory [n]    recent remembered exchanges
/teach <text>  learn from a passage directly
/upload <file> [as <name>]   read a document or transcript and learn from it
                             (or drag the file into the terminal)
/correct <text>  replace Aria's last reply with yours and learn from it (weight 3x, bypasses the gate)
/grow [n]      add n layers (default 1) without losing anything learned
/brain         a map of her brain: cortex, areas, hippocampus, sleep, novelty
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
- **Online learning is not memory, and memory is not understanding.** A
  gradient step changes a *disposition*, not a retrievable fact. The
  hippocampus does keep facts after one hearing, but the small cortex can
  rarely turn a recalled fact into an answer ([details](#the-hippocampus-one-shot-memory)).
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
- **Sized for a CPU.** A GPU is used automatically when present, but the presets are sized for
  a CPU budget.

## Layout

```
aria/
  config.py      dataclass configs and size presets
  tokenizer.py   byte-level BPE, trained from scratch
  model.py       the transformer, cortical areas, KV cache, LoRA attach/merge, growth
  data.py        corpus prep, chat formatting, batching
  pretrain.py    offline training loop + Fisher estimation
  learner.py     the online learning engine
  memory.py      replay buffer and decision journal (bounded)
  documents.py   streaming readers for uploads: txt/md/docx/srt/vtt/pdf, transcripts
  storage.py     compact, atomic weight files (float16, int8, log-encoded Fisher)
  device.py      choosing NVIDIA GPU / Apple GPU / CPU, with a self-test
  hippocampus.py fast, one-shot episodic memory
  optim.py       a low-memory Adam for large models
  sample.py      generation
  chat.py        REPL
  serve.py       local browser UI and background learning (standard library only)
  cli.py         command line
  seed_dialogues.txt        hand-written conversation seed
tests/           213 tests
notebooks/       Colab notebook for a free GPU
scripts/demo_learning.py     measures whether the learning actually works
scripts/recompute_fisher.py  re-estimates a checkpoint's Fisher information
```
