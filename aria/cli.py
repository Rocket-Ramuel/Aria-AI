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
        offline=args.offline,
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


def _learner_overrides(args):
    from .config import LearnerConfig
    cfg = LearnerConfig()
    for field in dataclasses.fields(cfg):
        val = getattr(args, f"learner_{field.name}", None)
        if val is not None:
            setattr(cfg, field.name, val)
    return cfg


def _cmd_chat(args) -> int:
    from .chat import ChatSession, run
    torch.set_num_threads(args.threads)
    session = ChatSession(
        checkpoint=args.checkpoint,
        state_dir=args.state_dir,
        data_dir=args.data_dir,
        learner_cfg=_learner_overrides(args),
        device=args.device,
        learning=not args.no_learn,
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
        learner_cfg=_learner_overrides(args),
        device=args.device,
        learning=not args.no_learn,
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


def _cmd_sample(args) -> int:
    from .pretrain import load_checkpoint
    from .sample import complete
    torch.set_num_threads(args.threads)
    model, tok, _, _ = load_checkpoint(args.checkpoint, args.device)
    from .learner import resume_learned_weights
    if args.state_dir:
        resume_learned_weights(model, args.state_dir, args.device)
    print(args.prompt, end="")
    print(complete(model, tok, args.prompt, device=args.device,
                   max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature))
    return 0


def _cmd_status(args) -> int:
    from .memory import Journal, ReplayBuffer
    state = Path(args.state_dir)
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
    """Batch-teach from a text file, one passage per line or per paragraph."""
    from .chat import ChatSession
    torch.set_num_threads(args.threads)
    session = ChatSession(
        checkpoint=args.checkpoint, state_dir=args.state_dir,
        data_dir=args.data_dir, device=args.device, learning=True,
    )
    text = Path(args.file).read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 40]
    for i, chunk in enumerate(chunks[: args.limit], 1):
        r = session.learner.observe_text(chunk, weight=args.weight)
        print(f"[{i}/{min(len(chunks), args.limit)}] {r.line()}")
    session.save()
    print(f"saved to {session.state_dir}")
    return 0


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
    ch.add_argument("--checkpoint", default="runs/aria/base.pt")
    ch.add_argument("--state-dir", default=None,
                    help="where learned weights and memory live (default <ckpt dir>/online)")
    ch.add_argument("--data-dir", default="data")
    ch.add_argument("--device", default="cpu")
    ch.add_argument("--no-learn", action="store_true", help="talk without updating weights")
    ch.add_argument("--verbose", action="store_true", help="print learner diagnostics each turn")
    ch.add_argument("--max-new-tokens", type=int, default=96)
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
    sv.add_argument("--checkpoint", default="runs/aria/base.pt")
    sv.add_argument("--state-dir", default=None)
    sv.add_argument("--data-dir", default="data")
    sv.add_argument("--host", default="127.0.0.1",
                    help="bind address; leave as localhost unless you mean it")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--no-browser", action="store_true")
    sv.add_argument("--no-learn", action="store_true")
    sv.add_argument("--max-new-tokens", type=int, default=96)
    sv.add_argument("--temperature", type=float, default=0.85)
    sv.add_argument("--device", default="cpu")
    _add_learner_flags(sv)
    sv.set_defaults(func=_cmd_serve)

    sm = sub.add_parser("sample", parents=[common], help="free-form completion from a prompt")
    sm.add_argument("--checkpoint", default="runs/aria/base.pt")
    sm.add_argument("--state-dir", default=None)
    sm.add_argument("--prompt", default="The ")
    sm.add_argument("--max-new-tokens", type=int, default=120)
    sm.add_argument("--temperature", type=float, default=0.85)
    sm.add_argument("--device", default="cpu")
    sm.set_defaults(func=_cmd_sample)

    st = sub.add_parser("status", parents=[common], help="report what the learner has been doing")
    st.add_argument("--state-dir", default="runs/aria/online")
    st.set_defaults(func=_cmd_status)

    te = sub.add_parser("teach", parents=[common], help="learn from a text file, paragraph by paragraph")
    te.add_argument("file")
    te.add_argument("--checkpoint", default="runs/aria/base.pt")
    te.add_argument("--state-dir", default=None)
    te.add_argument("--data-dir", default="data")
    te.add_argument("--device", default="cpu")
    te.add_argument("--limit", type=int, default=500)
    te.add_argument("--weight", type=float, default=2.0)
    te.set_defaults(func=_cmd_teach)

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
