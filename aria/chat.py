"""The conversational REPL — the thing you actually run.

Each turn: build the chat prompt, stream a reply, then hand the completed
exchange to the online learner, which decides whether to take a gradient step.
"""

from __future__ import annotations

import dataclasses
import shlex
import sys
from pathlib import Path
from typing import Any

from .config import LearnerConfig
from .data import TokenStream
from .documents import (extract_text, parse_transcript, speakers,
                        strip_speaker_labels, transcript_dialogues)
from .learner import OnlineLearner, Progress, UpdateReport, resume_learned_weights
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
                         learn from a whole document (.txt .md .docx .srt .vtt .pdf);
                         for a transcript, "as <name>" makes Aria answer like <name>
  /correct <text>        replace Aria's last reply with <text> and learn from it
  /learn on|off          enable or disable online learning
  /verbose on|off        show the learner's per-turn diagnostics
  /temp <float>          sampling temperature
  /consolidate           force a consolidation pass now
  /forget                clear the replay buffer (weights are untouched)
  /save                  write learned weights and memory to disk
  /reset                 clear the current conversation context
  /quit                  save and exit
""".strip()


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
        device: str = "cpu",
        learning: bool = True,
        verbose: bool = False,
        max_new_tokens: int | None = None,
        temperature: float = 0.85,
        top_k: int = 40,
        top_p: float = 0.92,
        learner_overrides: dict[str, Any] | None = None,
        blank: bool = False,
    ) -> None:
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

        self.learner = OnlineLearner(
            self.model, self.tok, cfg,
            fisher=ckpt.get("fisher"), pretrain_stream=stream,
            state_dir=self.state_dir, device=device,
        )
        self.history: list[tuple[str, str]] = []
        self.restored = restored

    # ------------------------------------------------------------------

    def reply(self, user_message: str, stream_to=None) -> str:
        prompt = build_chat_prompt(self.tok, self.history, user_message,
                                   self.model.cfg.block_size)
        pieces: list[int] = []
        decoder = self.tok.stream_decoder()
        for tid in generate(
            self.model, prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            stop_ids=(self.tok.eot_id, self.tok.user_id, self.tok.bos_id),
            device=self.device,
        ):
            pieces.append(tid)
            if stream_to is not None:
                stream_to.write(decoder.feed(tid))
                stream_to.flush()
        if stream_to is not None:
            stream_to.write(decoder.flush())
        return self.tok.decode(pieces, skip_special=True).strip()

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

    def upload(
        self,
        name: str,
        data: bytes,
        speaker: str | None = None,
        passes: int | None = None,
        weight: float = 1.0,
        progress: Progress = None,
    ) -> tuple[UpdateReport, str]:
        """Learn from a whole file. Returns the report and a one-line summary.

        A transcript with `speaker` named is learned as conversations in which
        that person plays Aria. Anything else is learned as prose in Aria's
        voice — for a transcript, with the speaker labels taken out so she
        learns the speech and not the formatting."""
        text = extract_text(name, data)
        turns = parse_transcript(text)
        if turns and speaker:
            names = speakers(turns)
            if not any(speaker.casefold() == n.casefold() for n in names):
                raise ValueError(f"nobody called {speaker!r} speaks in {name}; "
                                 f"speakers are: {', '.join(names[:8])}")
            convos = transcript_dialogues(turns, speaker)
            report = self.learner.learn_dialogues(convos, passes=passes, weight=weight,
                                                  name=name, progress=progress)
            what = f"{len(convos)} replies by {speaker}"
        else:
            if turns:
                text = strip_speaker_labels(turns)
            report = self.learner.learn_document(text, passes=passes, weight=weight,
                                                 name=name, progress=progress)
            what = f"{len(text.split())} words"
            if turns:
                what += (f" (a transcript: name a speaker to learn how one person "
                         f"replies — speakers: {', '.join(speakers(turns)[:5])})")
        # Learned material should survive a crash as surely as a chat turn.
        self.save()
        return report, f"learned {name}: {what}"

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

        if line.startswith("/"):
            if _command(session, line):
                break
            continue

        print("aria> ", end="", flush=True)
        _, report = session.turn(line, stream_to=sys.stdout)
        print()
        if report is not None and session.verbose:
            print(report.line())
        print()

    session.save()
    print("saved. goodbye.")


def _command(session: ChatSession, line: str) -> bool:
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
        try:
            words = shlex.split(arg)
        except ValueError as e:
            words = []
            print(f"  {e}")
        speaker = None
        if len(words) >= 3 and words[-2].lower() == "as":
            speaker, words = words[-1], words[:-2]
        if len(words) != 1:
            print("  usage: /upload <file> [as <speaker name>]")
        else:
            path = Path(words[0]).expanduser()
            try:
                report, summary = session.upload(
                    path.name, path.read_bytes(), speaker=speaker,
                    progress=_print_progress)
                print(f"  {summary}\n  {report.line()}")
            except (OSError, ValueError) as e:
                print(f"  can't learn from {path}: {e}")
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
    elif cmd == "/consolidate":
        session.learner.consolidate()
        print("  consolidated;", f"canary {session.learner.canary_loss():.4f}")
    elif cmd == "/forget":
        session.learner.replay.items.clear()
        session.learner.replay.seen = 0
        print("  replay buffer cleared (weights unchanged)")
    elif cmd == "/save":
        session.save()
        print(f"  saved to {session.state_dir}")
    elif cmd == "/reset":
        session.history.clear()
        print("  conversation context cleared")
    else:
        print(f"  unknown command {cmd}; try /help")
    return False


def _print_progress(step: int, total: int, lr: float) -> None:
    print(f"\r  learning ... step {step}/{total}", end="\n" if step == total else "",
          flush=True)
