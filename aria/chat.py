"""The conversational REPL — the thing you actually run.

Each turn: build the chat prompt, stream a reply, then hand the completed
exchange to the online learner, which decides whether to take a gradient step.
"""

from __future__ import annotations

import dataclasses
import itertools
import shlex
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Union

from .config import LearnerConfig
from .data import TokenStream
from .documents import (SUPPORTED_SUFFIXES, iter_dialogues, iter_lines, iter_turns,
                        iter_units, parse_transcript, speakers, suffix_of)
from .learner import (LearnProgress, OnlineLearner, Progress, StopCheck, UpdateReport,
                      drive, resume_learned_weights)
from .model import grow, merge_lora
from .storage import process_ram_mb
from .pretrain import (BLANK_CHECKPOINT, create_blank_checkpoint, load_checkpoint,
                       resolve_checkpoint)
from .sample import build_chat_prompt, generate
from .tokenizer import BPETokenizer

HELP = """
commands:
  /help                  this message
  /status                learner and memory statistics
  /memory [n]            show the n most recent remembered exchanges
  /teach <text>          learn from a passage directly (no reply generated)
  /upload <file> [as <name>]
                         read a document and learn from it (.txt .md .docx .srt .vtt
                         .pdf), any size; Ctrl-C stops early and keeps what she learned.
                         For a transcript, "as <name>" makes Aria answer like <name>.
                         Dragging a file into the terminal does the same.
  /correct <text>        replace Aria's last reply with <text> and learn from it
  /learn on|off          enable or disable online learning
  /verbose on|off        show the learner's per-turn diagnostics
  /temp <float>          sampling temperature
  /consolidate           force a consolidation pass now
  /grow [n]              add n layers (default 1): more room to learn, kept knowledge
  /brain                 a map of her brain: cortex, areas, hippocampus, sleep
  /forget                clear the replay buffer (weights are untouched)
  /save                  write learned weights and memory to disk
  /reset                 clear the current conversation context
  /quit                  save and exit
""".strip()


# How much of a file is read to decide whether it is a transcript.
SNIFF_LINES = 400
# During a long upload, what has been learned so far is saved this often, so a
# crash or a closed laptop loses minutes, not hours.
AUTOSAVE_SECONDS = 300

FileSource = Union[str, Path, bytes]


def default_state_dir(checkpoint: str | Path) -> Path:
    """Where a checkpoint's learned weights and memories live by default.

    Next to the checkpoint, so two different base models never share (and
    corrupt) one memory. Every command that touches learner state resolves it
    through here, so they all agree on where to look."""
    return Path(checkpoint).parent / "online"


def resolve_session_checkpoint(checkpoint: str | Path | None, blank: bool) -> Path:
    if blank and not checkpoint:
        if not BLANK_CHECKPOINT.exists():
            create_blank_checkpoint(BLANK_CHECKPOINT)
            print(f"created a blank model at {BLANK_CHECKPOINT}: it knows no "
                  f"words yet, so teach it with /upload or `aria teach`.")
        return BLANK_CHECKPOINT
    return resolve_checkpoint(checkpoint)


def _matching_corpus(data_dir: Path, tok: BPETokenizer, block_size: int):
    """The pretraining stream for rehearsal, only if it was encoded with this
    model's tokenizer — token ids from a different vocabulary are noise at
    best and an index error at worst."""
    train_bin, tok_json = data_dir / "train.bin", data_dir / "tokenizer.json"
    if not (train_bin.exists() and tok_json.exists()):
        return None
    if [tuple(m) for m in BPETokenizer.load(tok_json).merges] != tok.merges:
        return None
    return TokenStream(train_bin, block_size)


class ChatSession:
    def __init__(
        self,
        checkpoint: str | Path | None = None,
        state_dir: str | Path | None = None,
        data_dir: str | Path = "data",
        learner_cfg: LearnerConfig | None = None,
        device: str = "auto",
        learning: bool = True,
        verbose: bool = False,
        max_new_tokens: int | None = None,
        temperature: float = 0.85,
        top_k: int = 40,
        top_p: float = 0.92,
        learner_overrides: dict[str, Any] | None = None,
        blank: bool = False,
    ) -> None:
        from .device import resolve
        device = resolve(device)
        checkpoint = resolve_session_checkpoint(checkpoint, blank)
        self.checkpoint = checkpoint
        self.state_dir = Path(state_dir or default_state_dir(checkpoint))
        self.device = device
        self.learning = learning
        self.verbose = verbose
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.model, self.tok, self.cfg, ckpt = load_checkpoint(checkpoint, device)
        # Layers the checkpoint came with; any above this were grown.
        self.base_layers = self.model.cfg.n_layer
        # Text the checkpoint was already trained on counts toward how full
        # it is (see OnlineLearner.capacity_tokens).
        t = self.cfg.train
        self.pretrain_tokens = int(ckpt.get("step") or 0) * t.batch_size * t.grad_accum \
            * self.cfg.model.block_size
        # A byte-level model spends several tokens per word.
        self.max_new_tokens = max_new_tokens or (96 if self.tok.merges else 240)
        restored = resume_learned_weights(self.model, self.state_dir, device)

        # Command-line overrides apply on top of the config the checkpoint
        # carries, so a blank model keeps its blank-model learner settings.
        cfg = learner_cfg or dataclasses.replace(self.cfg.learner,
                                                 **(learner_overrides or {}))
        stream = None
        if cfg.pretrain_replay_frac > 0:
            stream = _matching_corpus(Path(data_dir), self.tok, self.model.cfg.block_size)

        self.device = device
        self._learner_cfg, self._stream = cfg, stream
        # Kept only where EWC uses it, for rebuilding the learner after growth.
        self._fisher = ckpt.get("fisher") if cfg.plasticity != "lora" else None
        self.learner = OnlineLearner(
            self.model, self.tok, cfg,
            fisher=ckpt.get("fisher"), pretrain_stream=stream,
            state_dir=self.state_dir, device=device,
            grown_from=self.base_layers, prior_tokens=self.pretrain_tokens,
        )
        del ckpt
        self.busy = False          # an upload is in progress
        self.history: list[tuple[str, str]] = []
        self.restored = restored

    # ------------------------------------------------------------------

    def reply(self, user_message: str, stream_to=None) -> str:
        prompt = build_chat_prompt(self.tok, self.history, user_message,
                                   self.model.cfg.block_size)
        pieces: list[int] = []
        decoder = self.tok.stream_decoder()
        memory = self.learner.hippocampus
        if memory is not None:
            memory.last_recall = None
        with self.learner.fast():
            for tid in generate(
                self.model, prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                stop_ids=(self.tok.eot_id, self.tok.user_id, self.tok.bos_id),
                device=self.device, memory=memory,
            ):
                pieces.append(tid)
                if stream_to is not None:
                    stream_to.write(decoder.feed(tid))
                    stream_to.flush()
        if stream_to is not None:
            stream_to.write(decoder.flush())
        # What the hippocampus contributed to this reply, if anything.
        self.last_recall = memory.last_recall if memory is not None else None
        return self.tok.decode(pieces, skip_special=True).strip()

    def recall_line(self) -> str | None:
        if not getattr(self, "last_recall", None):
            return None
        sim, what = self.last_recall
        return f"[recall] remembered \u201c{what}\u201d (match {sim:.2f})"

    def turn(self, user_message: str, stream_to=None):
        text = self.reply(user_message, stream_to=stream_to)
        self.history.append(("user", user_message))
        self.history.append(("aria", text))
        report = None
        if self.learning:
            report = self.learner.observe(self._recent_turns())
        return text, report

    def _recent_turns(self, n_turns: int = 4) -> list[str]:
        """The tail of the conversation, as alternating strings starting with
        the user. Including a little context means the learner sees the reply
        in the situation that produced it."""
        tail = self.history[-n_turns * 2:]
        while tail and tail[0][0] != "user":
            tail = tail[1:]
        return [t for _, t in tail]

    def correct(self, corrected_reply: str):
        """Replace Aria's last reply with the user's version and learn from it.

        Corrections carry extra weight and bypass the surprise gate, because
        being told "no, say it like this" is the highest-value signal available.
        """
        if len(self.history) < 2 or self.history[-1][0] != "aria":
            return None
        self.history[-1] = ("aria", corrected_reply)
        return self.learner.observe(self._recent_turns(), weight=3.0,
                                    kind="correction", force=True)

    def sniff(self, source: FileSource, name: str) -> tuple[list[str] | None, bool]:
        """(speakers if `source` is a transcript, whether all of it was read).

        Only the first SNIFF_LINES lines are read, so this is cheap for a
        file of any size."""
        head = list(itertools.islice(iter_lines(source, name), SNIFF_LINES + 1))
        if not any(line.strip() for line in head):
            raise ValueError(f"{name} contains no text")
        turns = parse_transcript(head[:SNIFF_LINES])
        return (speakers(turns) if turns else None), len(head) <= SNIFF_LINES

    def iter_learn_file(
        self,
        source: FileSource,
        name: str | None = None,
        speaker: str | None = None,
        passes: int | None = None,
        weight: float = 1.0,
        should_stop: StopCheck = None,
    ) -> Generator[LearnProgress, None, tuple[UpdateReport, str]]:
        """Read a document and learn from it, one step per iteration.

        `source` is a path (any size: it is streamed, never loaded whole) or
        the bytes of a small file. A transcript with `speaker` named is
        learned as conversations in which that person plays Aria; anything
        else as prose in Aria's voice — for a transcript, with the speaker
        labels taken out so she learns the speech and not the formatting.
        Returns the report and a one-line summary."""
        name = name or Path(str(source)).name
        names, whole = self.sniff(source, name)

        def lines():
            return iter_lines(source, name)

        if names and speaker:
            if whole and not any(speaker.casefold() == n.casefold() for n in names):
                raise ValueError(f"nobody called {speaker!r} speaks in {name}; "
                                 f"speakers are: {', '.join(names[:8])}")
            gen = self.learner.iter_learn_dialogues(
                lambda: iter_dialogues(iter_turns(lines()), speaker),
                passes=passes, weight=weight, name=name, should_stop=should_stop)
        else:
            if names:
                make_units = lambda: iter_units(said for _, said in iter_turns(lines()))
            else:
                make_units = lambda: iter_units(lines())
            gen = self.learner.iter_learn_units(make_units, passes=passes, weight=weight,
                                                name=name, should_stop=should_stop)

        last_save = time.monotonic()
        self.busy = True
        try:
            report = yield from self._drive_with_autosave(gen, last_save)
        finally:
            self.busy = False
        # Learned material should survive a crash as surely as a chat turn.
        self.save()
        grew = self.maybe_grow() if report.applied and not report.stopped else ""

        if not report.applied:
            if names and speaker:
                raise ValueError(f"nobody called {speaker!r} answers anyone in {name}; "
                                 f"speakers are: {', '.join(names[:8])}")
            return report, f"nothing learned from {name}: {report.reason}"
        if names and speaker:
            what = f"{report.examples:,} replies by {speaker}"
        else:
            what = f"{report.words:,} words"
            if names:
                what += (f" (a transcript: name a speaker to learn how one person "
                         f"replies — speakers: {', '.join(names[:5])})")
        return report, f"learned {name}: {what}" + (f"\n{grew}" if grew else "")

    def _drive_with_autosave(self, gen, last_save):
        while True:
            try:
                p = next(gen)
            except StopIteration as done:
                return done.value
            if time.monotonic() - last_save > AUTOSAVE_SECONDS:
                self.save()
                last_save = time.monotonic()
            yield p

    # -- growth ----------------------------------------------------------

    def grow(self, n_layers: int = 1) -> str:
        """Add `n_layers` layers on top, keeping everything learned.

        The new layers start as exact no-ops (see `aria.model.grow`), so
        nothing she says changes until they learn. Adapters are folded into
        the weights first, through a normal consolidation so the canary still
        gets its say; if it refuses, the unhealthy adapters are discarded."""
        if self.busy:
            raise RuntimeError("she is reading a document; grow after it finishes")
        before_layers = self.model.cfg.n_layer
        before_mb = self.model.num_params() * 2 / 1e6
        if self.learner.cfg.plasticity == "lora" and not self.learner.consolidate():
            self.learner._restore(self.learner.last_good)
        self.learner.save(weights=False)
        merge_lora(self.model)
        grow(self.model, n_layers)
        self.learner = OnlineLearner(
            self.model, self.tok, self._learner_cfg, fisher=self._fisher,
            pretrain_stream=self._stream, state_dir=self.state_dir, device=self.device,
            grown_from=self.base_layers, prior_tokens=self.pretrain_tokens,
        )
        self.learner.journal.write(event="grow", layers_before=before_layers,
                                   layers_after=self.model.cfg.n_layer)
        self.save()
        after_mb = self.model.num_params() * 2 / 1e6
        return (f"grew from {before_layers} to {self.model.cfg.n_layer} layers "
                f"({before_mb:.1f} MB -> {after_mb:.1f} MB of weights)")

    def maybe_grow(self) -> str:
        """Grow by one layer if she has learned from more text than her size
        has room for, up to `grow_max_factor` times her original depth."""
        cfg = self.learner.cfg
        limit = int(self.base_layers * cfg.grow_max_factor)
        if cfg.grow_max_factor <= 0 or self.learner.room_left() > 0 \
                or self.model.cfg.n_layer >= limit:
            return ""
        return "she has read more than her size has room for, so she " + self.grow(1)

    def learn_file(self, source: FileSource, name: str | None = None,
                   speaker: str | None = None, passes: int | None = None,
                   weight: float = 1.0, progress: Progress = None,
                   should_stop: StopCheck = None) -> tuple[UpdateReport, str]:
        """`iter_learn_file`, run to the end."""
        return drive(self.iter_learn_file(source, name, speaker, passes, weight,
                                          should_stop), progress)

    def upload(self, name: str, data: bytes, speaker: str | None = None,
               passes: int | None = None, weight: float = 1.0,
               progress: Progress = None) -> tuple[UpdateReport, str]:
        """Learn from the bytes of a file (for small ones; large ones are
        learned from disk with `learn_file`)."""
        return self.learn_file(data, name, speaker, passes, weight, progress)

    def save(self) -> None:
        self.learner.save()


# ----------------------------------------------------------------------
# REPL
# ----------------------------------------------------------------------


def run(session: ChatSession, banner: bool = True) -> None:
    if banner:
        n = session.model.num_params() / 1e6
        print(f"Aria — {n:.1f}M parameters, plasticity={session.learner.cfg.plasticity}, "
              f"learning={'on' if session.learning else 'off'}")
        if session.restored:
            print(f"restored weights learned in earlier sessions "
                  f"({session.learner.updates_applied} updates so far)")
        print("type /help for commands, /quit to leave\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        dropped = _dropped_file(line)
        if dropped is not None:
            line = "/upload " + shlex.quote(str(dropped))
        if line.startswith("/"):
            if _command(session, line, interactive=True):
                break
            continue

        print("aria> ", end="", flush=True)
        _, report = session.turn(line, stream_to=sys.stdout)
        print()
        if session.verbose and session.recall_line():
            print(session.recall_line())
        if report is not None and session.verbose:
            print(report.line())
        print()

    session.save()
    print("saved. goodbye.")


def _dropped_file(line: str) -> Path | None:
    """A file dragged into the terminal arrives as its (often quoted) path."""
    try:
        words = shlex.split(line)
    except ValueError:
        return None
    if len(words) != 1:
        return None
    raw = words[0]
    if "/" not in raw and "\\" not in raw and raw == line:
        return None             # a bare word is a message, even if a file has that name
    path = Path(raw).expanduser()
    if suffix_of(path.name) in SUPPORTED_SUFFIXES and path.is_file():
        return path
    return None


@contextmanager
def _ctrl_c_stops():
    """Make Ctrl-C end a long upload cleanly instead of killing the program."""
    stop = threading.Event()
    if threading.current_thread() is not threading.main_thread():
        yield stop
        return
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        yield stop
    finally:
        signal.signal(signal.SIGINT, previous)


def _upload_command(session: ChatSession, arg: str, interactive: bool) -> None:
    try:
        words = shlex.split(arg)
    except ValueError as e:
        print(f"  {e}")
        return
    speaker = None
    if len(words) >= 3 and words[-2].lower() == "as":
        speaker, words = words[-1], words[:-2]
    if len(words) != 1:
        print("  usage: /upload <file> [as <speaker name>]")
        return
    path = Path(words[0]).expanduser()
    try:
        names, _ = session.sniff(path, path.name)
        if names and speaker is None and interactive:
            print(f"  {path.name} looks like a conversation between "
                  f"{', '.join(names[:6])}.")
            answer = input("  learn to reply like who? (a name, or Enter to just "
                           "learn the language) ").strip()
            speaker = answer or None
        size = path.stat().st_size
        print(f"  reading {path.name} ({size / 1e6:.1f} MB) — Ctrl-C stops early "
              f"and keeps what she has learned")
        with _ctrl_c_stops() as stop:
            report, summary = session.learn_file(path, path.name, speaker=speaker,
                                                 progress=_print_progress,
                                                 should_stop=stop.is_set)
        print(f"\r  {summary}".ljust(60))
        print(f"  {report.line()}")
    except (OSError, ValueError) as e:
        print(f"\n  can't learn from {path}: {e}")


def _command(session: ChatSession, line: str, interactive: bool = False) -> bool:
    """Handle a slash command. Returns True if the session should end."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return True
    if cmd == "/help":
        print(HELP)
    elif cmd == "/status":
        st = session.learner.status()
        width = max(len(k) for k in st)
        for k, v in st.items():
            v = f"{v:.4f}" if isinstance(v, float) else v
            print(f"  {k.ljust(width)}  {v}")
        print(f"  {'journal'.ljust(width)}  {session.learner.journal.summary()}")
        ram = process_ram_mb()
        if ram is not None:
            print(f"  {'peak_ram_mb'.ljust(width)}  {ram:.0f}")
    elif cmd == "/memory":
        n = int(arg) if arg.isdigit() else 8
        for item in session.learner.replay.recent(n):
            head = " | ".join(t[:60] for t in item["turns"][:2])
            print(f"  [{item['kind']}, w={item['weight']:.1f}] {head}")
    elif cmd == "/teach":
        if not arg:
            print("  usage: /teach <text>")
        else:
            print(" ", session.learner.observe_text(arg, weight=2.0).line())
    elif cmd == "/upload":
        _upload_command(session, arg, interactive)
    elif cmd == "/correct":
        if not arg:
            print("  usage: /correct <what Aria should have said>")
        else:
            r = session.correct(arg)
            print(" ", r.line() if r else "nothing to correct yet")
    elif cmd == "/learn":
        session.learning = arg.lower() not in ("off", "0", "false")
        print(f"  online learning {'on' if session.learning else 'off'}")
    elif cmd == "/verbose":
        session.verbose = arg.lower() not in ("off", "0", "false")
        print(f"  verbose {'on' if session.verbose else 'off'}")
    elif cmd == "/temp":
        try:
            session.temperature = float(arg)
            print(f"  temperature {session.temperature}")
        except ValueError:
            print("  usage: /temp 0.85")
    elif cmd == "/brain":
        print(brain_map(session))
    elif cmd == "/grow":
        n = int(arg) if arg.isdigit() and int(arg) > 0 else 1
        try:
            print(f"  {session.grow(n)}")
        except RuntimeError as e:
            print(f"  {e}")
    elif cmd == "/consolidate":
        session.learner.consolidate()
        print("  consolidated;", f"canary {session.learner.canary_loss():.4f}")
    elif cmd == "/forget":
        session.learner.replay.items.clear()
        session.learner.replay.seen = 0
        session.learner.sleep()          # the hippocampus is built from it
        print("  memories cleared (weights unchanged)")
    elif cmd == "/save":
        session.save()
        print(f"  saved to {session.state_dir}")
    elif cmd == "/reset":
        session.history.clear()
        print("  conversation context cleared")
    else:
        print(f"  unknown command {cmd}; try /help")
    return False


def brain_map(session: ChatSession) -> str:
    """Each part of Aria's brain, what it does, and what it's doing now."""
    from .model import Areas
    L, m = session.learner, session.model
    j = L.journal.summary()
    out = [f"  cortex       {m.num_params() / 1e6:.2f}M weights in {m.cfg.n_layer} layers "
           f"({session.base_layers} original) — slow, general knowledge; "
           f"{100 * L.room_left():.0f}% room left"]
    areas = [b.ffn for b in m.blocks if isinstance(b.ffn, Areas)]
    if areas:
        out.append(f"  areas        {len(areas[0].areas)} specialist areas per layer, "
                   f"2 used per word; share of words each area has taken:")
        for i, a in enumerate(areas):
            total = float(a.usage.sum()) or 1.0
            bars = "  ".join(f"{float(u) / total:4.0%}" for u in a.usage)
            out.append(f"               layer {i + 1}: {bars}")
    h = L.hippocampus
    if h is not None:
        out.append(f"  hippocampus  {len(h.episodes)} remembered moments ({len(h):,} words, "
                   f"{h.memory_mb():.1f} MB) — one-shot memory; recalled "
                   f"{h.recalls:,} times")
    else:
        out.append("  hippocampus  off")
    out.append(f"  sleep        {j['consolidations']} consolidations, {j['rollbacks']} "
               f"rollbacks — learning made permanent, memories re-encoded")
    out.append(f"  novelty      {j['updates_skipped']} familiar exchanges skipped, "
               f"{j['updates_applied']} learned — surprise decides what is worth learning")
    return "\n".join(out)


def _print_progress(p: LearnProgress) -> None:
    if p.phase == "reading":
        msg = f"reading ... {p.examples:,} pieces so far"
    else:
        msg = f"learning ... step {p.step:,}/{p.total:,} ({100 * p.step / max(1, p.total):.0f}%)"
    print(f"\r  {msg}".ljust(60), end="", flush=True)
