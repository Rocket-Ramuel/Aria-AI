"""End-to-end tests: the whole pipeline, through the real CLI, on tiny configs.

These are slower than the unit tests but they are the ones that catch wiring
mistakes — a prepare/pretrain/chat path that only works when driven by hand is
not much use.
"""

import json
from pathlib import Path

import pytest
import torch

from aria.cli import main
from aria.data import prepare
from aria.chat import ChatSession, _command
from aria.pretrain import load_checkpoint

CORPUS = """The river ran past the old mill and turned east towards the sea.
She opened the window and listened to the rain falling on the roof.
Every morning the baker lit the oven before the sky began to lighten.
They walked together along the quiet road until the light began to fade.
The letter arrived on a Tuesday and nobody knew who had sent it.
He counted the coins twice and put them back into the wooden box.
Winter came early that year and the fields stayed white until March.
The children ran ahead, shouting, and the dog followed close behind.
A single lamp burned in the window of the house on the corner.
The train was late again and the platform filled slowly with people.
"""


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Run prepare + pretrain once; the tests below share the result."""
    root = tmp_path_factory.mktemp("aria")
    data = root / "data"
    (data / "raw").mkdir(parents=True)
    (data / "raw" / "corpus.txt").write_text(CORPUS * 120)
    (data / "seed_dialogues.txt").write_text(
        "U: hello\nA: hello there\n\nU: who are you\nA: I am Aria\n"
    )

    prepare(data_dir=data, vocab_size=400, block_size=64,
            n_surrogate_dialogues=300, offline=True, verbose=False)

    out = root / "run"
    rc = main([
        "pretrain", "--data-dir", str(data), "--out-dir", str(out),
        "--preset", "tiny", "--block-size", "64", "--steps", "30",
        "--batch-size", "4", "--warmup", "5", "--eval-interval", "15",
        "--checkpoint-interval", "1000", "--fisher-batches", "3",
    ])
    assert rc == 0
    return root, data, out


def test_prepare_produces_every_artifact(workspace):
    _, data, _ = workspace
    for name in ["tokenizer.json", "train.bin", "val.bin", "chat.pt", "meta.json"]:
        assert (data / name).exists(), name
    meta = json.loads((data / "meta.json").read_text())
    assert meta["train_tokens"] > 1000
    assert meta["chat_examples"] > 0
    assert meta["vocab_size"] <= 400


def test_pretrain_checkpoint_is_self_contained(workspace):
    _, _, out = workspace
    model, tok, cfg, ckpt = load_checkpoint(out / "base.pt")
    # A checkpoint carries its own tokenizer and config, so nothing else on
    # disk is needed to load it.
    assert tok.vocab_size == model.cfg.vocab_size
    assert ckpt["fisher"] is not None
    assert set(ckpt["fisher"]) <= set(dict(model.named_parameters()))
    assert ckpt["val_loss"] > 0


def test_training_actually_reduced_loss(workspace):
    _, _, out = workspace
    rows = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert len(rows) >= 2
    assert rows[-1]["val_loss"] < rows[0]["val_loss"]


def test_pretrain_resumes_from_latest(workspace, capsys):
    _, data, out = workspace
    rc = main(["pretrain", "--data-dir", str(data), "--out-dir", str(out),
               "--preset", "tiny", "--block-size", "64", "--steps", "33",
               "--batch-size", "4", "--warmup", "5", "--eval-interval", "50",
               "--fisher-batches", "2"])
    assert rc == 0
    assert "resumed from" in capsys.readouterr().out


def test_sample_runs_from_the_cli(workspace, capsys):
    _, _, out = workspace
    assert main(["sample", "--checkpoint", str(out / "base.pt"),
                 "--prompt", "The river", "--max-new-tokens", "12"]) == 0
    assert "The river" in capsys.readouterr().out


def make_session(workspace, **kw):
    root, data, out = workspace
    return ChatSession(checkpoint=out / "base.pt",
                       state_dir=root / "online", data_dir=data, **kw)


def test_chat_session_replies_and_learns(workspace):
    s = make_session(workspace)
    text, report = s.turn("hello there")
    assert isinstance(text, str)
    assert report is not None and report.loss_before > 0
    assert len(s.history) == 2
    assert s.history[0] == ("user", "hello there")


def test_reply_never_contains_special_tokens(workspace):
    s = make_session(workspace)
    for msg in ["hello", "who are you", "tell me about the river"]:
        text, _ = s.turn(msg)
        for special in s.tok.specials:
            assert special not in text


def test_no_learn_mode_leaves_weights_untouched(workspace):
    s = make_session(workspace, learning=False)
    before = {n: p.clone() for n, p in s.model.named_parameters()}
    for msg in ["hello", "what is the river", "tell me a story"]:
        s.turn(msg)
    for n, p in s.model.named_parameters():
        assert torch.equal(before[n], p), n


def test_correction_is_learned_with_extra_weight(workspace):
    s = make_session(workspace)
    s.turn("who are you")
    report = s.correct("I am Aria and I run on your own machine")
    assert report is not None and report.applied
    assert s.history[-1] == ("aria", "I am Aria and I run on your own machine")
    assert s.learner.replay.recent(1)[0]["kind"] == "correction"


def test_long_history_is_truncated_to_the_context_window(workspace):
    s = make_session(workspace, learning=False)
    s.history = [("user" if i % 2 == 0 else "aria", f"turn number {i} " * 12)
                 for i in range(80)]
    text, _ = s.turn("and what do you think about all that")
    assert isinstance(text, str)   # must not raise on an over-long context


def test_slash_commands_do_not_crash(workspace, capsys):
    s = make_session(workspace)
    s.turn("hello")
    for cmd in ["/help", "/status", "/memory 3", "/teach the mill is by the river",
                "/correct actually the mill is east of the river", "/learn off",
                "/learn on", "/verbose on", "/temp 0.7", "/consolidate",
                "/save", "/reset", "/nonsense"]:
        assert _command(s, cmd) is False
    assert _command(s, "/quit") is True
    out = capsys.readouterr().out
    assert "unknown command" in out
    assert not s.history          # /reset cleared it


def test_forget_clears_memory_but_not_weights(workspace):
    s = make_session(workspace)
    s.turn("hello")
    before = {n: p.clone() for n, p in s.model.named_parameters()}
    _command(s, "/forget")
    assert len(s.learner.replay) == 0
    for n, p in s.model.named_parameters():
        assert torch.equal(before[n], p)


def test_status_command_reports_state(workspace, capsys):
    root, _, _ = workspace
    s = make_session(workspace)
    s.turn("hello")
    s.save()
    assert main(["status", "--state-dir", str(root / "online")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["runtime"]["plasticity"] == "lora"
    assert out["journal"]["turns_seen"] >= 1


def test_teach_command_learns_from_a_file(workspace, tmp_path, capsys):
    root, data, out = workspace
    doc = tmp_path / "notes.txt"
    doc.write_text("The mill by the river was built in eighteen twelve.\n\n"
                   "It ground wheat for the whole valley until the war.\n")
    rc = main(["teach", str(doc), "--checkpoint", str(out / "base.pt"),
               "--state-dir", str(root / "teach_state"), "--data-dir", str(data)])
    assert rc == 0
    assert (root / "teach_state" / "learned.pt").exists()
    assert "[1/2]" in capsys.readouterr().out
