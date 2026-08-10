"""The conversational REPL — the thing you actually run.

Each turn: build the chat prompt, stream a reply, then hand the completed
exchange to the online learner, which decides whether to take a gradient step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch

from .config import LearnerConfig
from .data import TokenStream
from .learner import OnlineLearner, resume_learned_weights
from .memory import Journal
from .pretrain import load_checkpoint
from .sample import build_chat_prompt, generate

HELP = """
commands:
  /help                  this message
  /status                learner and memory statistics
  /memory [n]            show the n most recent remembered exchanges
  /teach <text>          learn from a passage directly (no reply generated)
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


class ChatSession:
    def __init__(
        self,
        checkpoint: str | Path,
        state_dir: str | Path | None = None,
        data_dir: str | Path = "data",
        learner_cfg: LearnerConfig | None = None,
        device: str = "cpu",
        learning: bool = True,
        verbose: bool = False,
        max_new_tokens: int = 96,
        temperature: float = 0.85,
        top_k: int = 40,
        top_p: float = 0.92,
    ) -> None:
        checkpoint = Path(checkpoint)
        self.state_dir = Path(state_dir or checkpoint.parent / "online")
        self.device = device
        self.learning = learning
        self.verbose = verbose
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.model, self.tok, self.cfg, ckpt = load_checkpoint(checkpoint, device)
        restored = resume_learned_weights(self.model, self.state_dir, device)

        stream = None
        train_bin = Path(data_dir) / "train.bin"
        if train_bin.exists():
            stream = TokenStream(train_bin, self.model.cfg.block_size)

        self.learner = OnlineLearner(
            self.model, self.tok, learner_cfg or self.cfg.learner,
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
                stream_to.write(self.tok.decode([tid], skip_special=True))
                stream_to.flush()
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
