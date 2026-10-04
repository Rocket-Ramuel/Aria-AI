"""Command line entrypoint: `python -m aria <command>`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch


def _cmd_prepare(args) -> int:
    from .data import prepare
    prepare(
        data_dir=args.data_dir,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_surrogate_dialogues=args.surrogate_dialogues,
        seed_repeat=args.seed_repeat,
        offline=args.offline,
        chat_only=args.chat_only,
    )
    return 0


def _cmd_pretrain(args) -> int:
    from .config import TrainConfig, preset
    from .pretrain import pretrain

    model_cfg = preset(args.preset)
    if args.block_size:
        model_cfg.block_size = args.block_size

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_steps=args.steps,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        max_minutes=args.max_minutes,
        fisher_batches=args.fisher_batches,
        warmup_steps=min(args.warmup, max(1, args.steps // 10)),
    )
    torch.set_num_threads(args.threads)
    pretrain(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        chat_frac=args.chat_frac,
        resume=not args.no_resume,
        device=args.device,
    )
    return 0


def _learner_overrides(args) -> dict:
    """Only the --learner-* flags actually given. They are applied on top of
    the learner config stored in the checkpoint, so a flag changes one setting
    instead of silently resetting the rest to the generic defaults."""
    from .config import LearnerConfig
    out = {}
    for field in dataclasses.fields(LearnerConfig):
        val = getattr(args, f"learner_{field.name}", None)
        if val is not None:
            out[field.name] = val
    return out


def _cmd_chat(args) -> int:
    from .chat import ChatSession, run
    torch.set_num_threads(args.threads)
    session = ChatSession(
        checkpoint=args.checkpoint,
        state_dir=args.state_dir,
        data_dir=args.data_dir,
        learner_overrides=_learner_overrides(args),
        device=args.device,
        learning=not args.no_learn,
        blank=args.blank,
        verbose=args.verbose,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    run(session)
    return 0


def _cmd_serve(args) -> int:
    from .serve import serve
    torch.set_num_threads(args.threads)
    serve(
        checkpoint=args.checkpoint,
        state_dir=args.state_dir,
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        allow_hosts=tuple(args.allow_host),
        learner_overrides=_learner_overrides(args),
        device=args.device,
        learning=not args.no_learn,
        blank=args.blank,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    return 0


def _cmd_quickstart(args) -> int:
    """prepare + pretrain + launch, with defaults chosen to finish on a laptop.

    This exists so that "get me something I can talk to" is one command rather
    than three with a page of flags between them.
    """
    from .config import TrainConfig, preset
    from .data import prepare
    from .pretrain import pretrain

    torch.set_num_threads(args.threads)
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    model_cfg = preset(args.preset)

    if (data_dir / "train.bin").exists() and not args.force_prepare:
        print(f"using the corpus already in {data_dir} "
              f"(pass --force-prepare to rebuild)")
    else:
        print("step 1/3  building the corpus and tokenizer ...")
        prepare(data_dir=data_dir, vocab_size=model_cfg.vocab_size,
                block_size=model_cfg.block_size, offline=args.offline)

    print(f"\nstep 2/3  pretraining for up to {args.minutes:g} minutes "
          f"(resumable: re-run to continue) ...")
    train_cfg = TrainConfig(
        batch_size=args.batch_size, max_steps=args.steps, learning_rate=args.lr,
        warmup_steps=min(200, max(1, args.steps // 10)), max_minutes=args.minutes,
        eval_interval=250, checkpoint_interval=250, fisher_batches=args.fisher_batches,
    )
    ckpt = pretrain(data_dir=data_dir, out_dir=out_dir, model_cfg=model_cfg,
                    train_cfg=train_cfg, device=args.device)

    if args.no_launch:
        print(f"\ndone. talk to her with:  aria chat --checkpoint {ckpt}")
        return 0

    print("\nstep 3/3  starting the chat UI ...")
    from .serve import serve
    serve(checkpoint=ckpt, data_dir=data_dir, host=args.host, port=args.port,
          open_browser=not args.no_browser, device=args.device)
    return 0


def _cmd_export(args) -> int:
    from .pretrain import export_checkpoint, resolve_checkpoint
    info = export_checkpoint(resolve_checkpoint(args.checkpoint), args.out,
                             half=not args.full_precision,
                             keep_fisher=not args.no_fisher)
    print(f"wrote {args.out}")
    print(f"  {info['source_mb']:.1f} MB -> {info['export_mb']:.1f} MB"
          f"  (half={info['half']}, fisher={info['fisher']})")
    val = info["val_loss"]
    val = f"{val:.4f}" if isinstance(val, float) else "n/a"
    print(f"  trained {info['step']} steps, val loss {val}")
    return 0


def _cmd_sample(args) -> int:
    from .pretrain import load_checkpoint, resolve_checkpoint
    from .sample import complete
    torch.set_num_threads(args.threads)
    model, tok, _, _ = load_checkpoint(resolve_checkpoint(args.checkpoint), args.device)
    from .learner import resume_learned_weights
    if args.state_dir:
        resume_learned_weights(model, args.state_dir, args.device)
    print(args.prompt, end="")
    print(complete(model, tok, args.prompt, device=args.device,
                   max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature))
    return 0


def _state_dir_for(args) -> Path:
    """The same default `chat` and `serve` use, so `status` looks where they
    actually wrote."""
    from .chat import default_state_dir
    from .pretrain import BLANK_CHECKPOINT, resolve_checkpoint
    if args.state_dir:
        return Path(args.state_dir)
    if args.blank and not args.checkpoint:
        return default_state_dir(BLANK_CHECKPOINT)
    return default_state_dir(resolve_checkpoint(args.checkpoint))


def _cmd_status(args) -> int:
    from .memory import Journal, ReplayBuffer
    state = _state_dir_for(args)
    journal = Journal(state / "journal.jsonl")
    replay = ReplayBuffer.load(state / "replay.json")
    runtime = state / "learner_state.json"
    out = {
        "state_dir": str(state),
        "replay_size": len(replay),
        "replay_seen": replay.seen,
        "journal": journal.summary(),
        "runtime": json.loads(runtime.read_text()) if runtime.exists() else None,
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_teach(args) -> int:
    """Learn from whole files: prose as Aria's voice, transcripts as replies."""
    from .chat import ChatSession, _print_progress
    torch.set_num_threads(args.threads)
    session = ChatSession(
        checkpoint=args.checkpoint, state_dir=args.state_dir,
        data_dir=args.data_dir, device=args.device, learning=True,
        blank=args.blank, learner_overrides=_learner_overrides(args),
    )
    failed = 0
    for f in args.files:
        path = Path(f)
        try:
            report, summary = session.upload(
                path.name, path.read_bytes(), speaker=args.speaker,
                passes=args.passes, weight=args.weight, progress=_print_progress)
        except (OSError, ValueError) as e:
            print(f"can't learn from {path}: {e}")
            failed += 1
            continue
        print(f"{summary}\n  {report.line()}")
    print(f"saved to {session.state_dir}")
    return 1 if failed else 0


def _cmd_blank(args) -> int:
    from .pretrain import create_blank_checkpoint
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists; pass --force to replace it "
              f"(its learned state in {out.parent / 'online'} is kept)")
        return 1
    create_blank_checkpoint(out, size=args.size, block_size=args.block_size)
    print(f"wrote a blank {args.size} model to {out}. It knows no words yet:")
    print(f"  aria teach --checkpoint {out} some_writing.txt")
    print(f"  aria serve --checkpoint {out}")
    return 0


BLANK_HELP = ("use a model with no pretraining (created at runs/blank/base.pt "
              "if needed) that learns only from what you upload and say")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("aria", description="A small English language model that keeps learning as you talk to it.")
    # Shared flags live on a parent parser so they are accepted *after* the
    # subcommand, which is where anyone would naturally type them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--threads", type=int, default=0,
                        help="torch CPU threads (0 = leave as-is)")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("prepare", parents=[common],
                        help="download the corpus, train the tokenizer, build datasets")
    pr.add_argument("--data-dir", default="data")
    pr.add_argument("--vocab-size", type=int, default=8192)
    pr.add_argument("--block-size", type=int, default=256)
    pr.add_argument("--surrogate-dialogues", type=int, default=40000)
    pr.add_argument("--seed-repeat", type=int, default=100,
                    help="how many times the hand-written conversation seed is "
                         "repeated relative to surrogate dialogues")
    pr.add_argument("--chat-only", action="store_true",
                    help="rebuild only chat.pt from the existing tokenizer and "
                         "corpus (fast; for retuning the conversation mix)")
    pr.add_argument("--offline", action="store_true",
                    help="use whatever .txt files are already in data/raw/")
    pr.set_defaults(func=_cmd_prepare)

    pt = sub.add_parser("pretrain", parents=[common], help="train the base model on the corpus")
    pt.add_argument("--data-dir", default="data")
    pt.add_argument("--out-dir", default="runs/aria")
    pt.add_argument("--preset", default="small", choices=["tiny", "small", "base"])
    pt.add_argument("--block-size", type=int, default=0)
    pt.add_argument("--steps", type=int, default=4000)
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--grad-accum", type=int, default=1)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--warmup", type=int, default=200)
    pt.add_argument("--eval-interval", type=int, default=200)
    pt.add_argument("--checkpoint-interval", type=int, default=250)
    pt.add_argument("--max-minutes", type=float, default=0.0,
                    help="wall-clock budget; training stops cleanly when reached")
    pt.add_argument("--fisher-batches", type=int, default=64)
    pt.add_argument("--chat-frac", type=float, default=0.25,
                    help="fraction of each batch drawn from chat-format examples")
    pt.add_argument("--no-resume", action="store_true")
    pt.add_argument("--device", default="cpu")
    pt.set_defaults(func=_cmd_pretrain)

    ch = sub.add_parser("chat", parents=[common], help="talk to Aria; she learns as you do")
    ch.add_argument("--checkpoint", default=None,
                   help="defaults to runs/aria/base.pt, then checkpoints/aria-small.pt")
    ch.add_argument("--state-dir", default=None,
                    help="where learned weights and memory live (default <ckpt dir>/online)")
    ch.add_argument("--data-dir", default="data")
    ch.add_argument("--device", default="cpu")
    ch.add_argument("--no-learn", action="store_true", help="talk without updating weights")
    ch.add_argument("--verbose", action="store_true", help="print learner diagnostics each turn")
    ch.add_argument("--blank", action="store_true", help=BLANK_HELP)
    ch.add_argument("--max-new-tokens", type=int, default=None,
                    help="reply length cap (default 96, or 240 for a byte-level blank model)")
    ch.add_argument("--temperature", type=float, default=0.85)
    _add_learner_flags(ch)
    ch.set_defaults(func=_cmd_chat)

    qs = sub.add_parser("quickstart", parents=[common],
                        help="one command: build the data, train, and open the chat UI")
    qs.add_argument("--data-dir", default="data")
    qs.add_argument("--out-dir", default="runs/aria")
    qs.add_argument("--preset", default="small", choices=["tiny", "small", "base"])
    qs.add_argument("--minutes", type=float, default=45.0,
                    help="wall-clock training budget; re-run to train further")
    qs.add_argument("--steps", type=int, default=5000)
    qs.add_argument("--batch-size", type=int, default=16)
    qs.add_argument("--lr", type=float, default=6e-4)
    qs.add_argument("--fisher-batches", type=int, default=64)
    qs.add_argument("--offline", action="store_true")
    qs.add_argument("--force-prepare", action="store_true")
    qs.add_argument("--no-launch", action="store_true")
    qs.add_argument("--no-browser", action="store_true")
    qs.add_argument("--host", default="127.0.0.1")
    qs.add_argument("--port", type=int, default=8000)
    qs.add_argument("--device", default="cpu")
    qs.set_defaults(func=_cmd_quickstart)

    sv = sub.add_parser("serve", parents=[common],
                        help="chat with Aria in a browser (local, no dependencies)")
    sv.add_argument("--checkpoint", default=None,
                   help="defaults to runs/aria/base.pt, then checkpoints/aria-small.pt")
    sv.add_argument("--state-dir", default=None)
    sv.add_argument("--data-dir", default="data")
    sv.add_argument("--host", default="127.0.0.1",
                    help="bind address; leave as localhost unless you mean it")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--no-browser", action="store_true")
    sv.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="extra hostname the page may be reached by (repeatable)")
    sv.add_argument("--no-learn", action="store_true")
    sv.add_argument("--blank", action="store_true", help=BLANK_HELP)
    sv.add_argument("--max-new-tokens", type=int, default=None)
    sv.add_argument("--temperature", type=float, default=0.85)
    sv.add_argument("--device", default="cpu")
    _add_learner_flags(sv)
    sv.set_defaults(func=_cmd_serve)

    sm = sub.add_parser("sample", parents=[common], help="free-form completion from a prompt")
    sm.add_argument("--checkpoint", default=None,
                   help="defaults to runs/aria/base.pt, then checkpoints/aria-small.pt")
    sm.add_argument("--state-dir", default=None)
    sm.add_argument("--prompt", default="The ")
    sm.add_argument("--max-new-tokens", type=int, default=120)
    sm.add_argument("--temperature", type=float, default=0.85)
    sm.add_argument("--device", default="cpu")
    sm.set_defaults(func=_cmd_sample)

    ex = sub.add_parser("export", parents=[common],
                        help="write a compact, shareable copy of a checkpoint")
    ex.add_argument("--checkpoint", default=None)
    ex.add_argument("--out", default="checkpoints/aria-small.pt")
    ex.add_argument("--full-precision", action="store_true",
                    help="keep float32 instead of halving the file size")
    ex.add_argument("--no-fisher", action="store_true",
                    help="drop the Fisher information (only needed for "
                         "--learner-plasticity ffn/full)")
    ex.set_defaults(func=_cmd_export)

    st = sub.add_parser("status", parents=[common], help="report what the learner has been doing")
    st.add_argument("--checkpoint", default=None)
    st.add_argument("--state-dir", default=None,
                    help="default: the 'online' directory next to the checkpoint")
    st.add_argument("--blank", action="store_true", help="the blank model's state")
    st.set_defaults(func=_cmd_status)

    te = sub.add_parser("teach", parents=[common],
                        help="learn from documents: writing samples, chat logs, transcripts")
    te.add_argument("files", nargs="+", metavar="FILE",
                    help=".txt .md .docx .srt .vtt (.pdf with pypdf installed)")
    te.add_argument("--speaker", default=None,
                    help="in a 'Name: words' transcript, learn to answer like this person")
    te.add_argument("--passes", type=int, default=None,
                    help="passes over each file (default from the learner config)")
    te.add_argument("--checkpoint", default=None,
                   help="defaults to runs/aria/base.pt, then checkpoints/aria-small.pt")
    te.add_argument("--blank", action="store_true", help=BLANK_HELP)
    te.add_argument("--state-dir", default=None)
    te.add_argument("--data-dir", default="data")
    te.add_argument("--device", default="cpu")
    te.add_argument("--weight", type=float, default=1.0)
    _add_learner_flags(te)
    te.set_defaults(func=_cmd_teach)

    bl = sub.add_parser("blank", parents=[common],
                        help="create a model that knows nothing and learns only from you")
    bl.add_argument("--out", default="runs/blank/base.pt")
    bl.add_argument("--size", default="small", choices=["tiny", "small", "base"])
    bl.add_argument("--block-size", type=int, default=512,
                    help="context window in bytes")
    bl.add_argument("--force", action="store_true")
    bl.set_defaults(func=_cmd_blank)

    return p


def _add_learner_flags(parser: argparse.ArgumentParser) -> None:
    """Expose every LearnerConfig field as --learner-<name>, defaulting to None
    so that unset flags fall through to the config defaults."""
    from .config import LearnerConfig
    g = parser.add_argument_group("online learning")
    for field in dataclasses.fields(LearnerConfig):
        flag = f"--learner-{field.name.replace('_', '-')}"
        if field.type is bool or isinstance(field.default, bool):
            g.add_argument(flag, dest=f"learner_{field.name}",
                           type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                           default=None, metavar="BOOL")
        elif isinstance(field.default, int) and not isinstance(field.default, bool):
            g.add_argument(flag, dest=f"learner_{field.name}", type=int, default=None)
        elif isinstance(field.default, float):
            g.add_argument(flag, dest=f"learner_{field.name}", type=float, default=None)
        else:
            g.add_argument(flag, dest=f"learner_{field.name}", type=str, default=None)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "threads", 0):
        torch.set_num_threads(args.threads)
    else:
        args.threads = torch.get_num_threads()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
