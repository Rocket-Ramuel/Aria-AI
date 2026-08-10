"""Tests for checkpoint resolution and the compact export.

The export exists so a trained model can be committed and used straight away.
That only helps if the exported file loads into exactly the same behaviour, so
that is what these check.
"""

import pytest
import torch

from aria.cli import main
from aria.data import prepare
from aria.model import GPT
from aria.pretrain import (
    DEFAULT_CHECKPOINTS, export_checkpoint, load_checkpoint, resolve_checkpoint,
)

CORPUS = """The river ran past the old mill, and turned east towards the sea.
She opened the window, and listened to the rain falling on the roof.
Every morning the baker lit the oven, long before the sky began to lighten.
"""


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt")
    data = root / "data"
    (data / "raw").mkdir(parents=True)
    (data / "raw" / "c.txt").write_text(CORPUS * 150)
    prepare(data_dir=data, vocab_size=400, block_size=64,
            n_surrogate_dialogues=150, offline=True, verbose=False)
    out = root / "run"
    assert main(["pretrain", "--data-dir", str(data), "--out-dir", str(out),
                 "--preset", "tiny", "--block-size", "64", "--steps", "20",
                 "--batch-size", "4", "--warmup", "5", "--eval-interval", "20",
                 "--checkpoint-interval", "1000", "--fisher-batches", "2"]) == 0
    return root, out / "base.pt"


# --- resolution ------------------------------------------------------------


def test_explicit_path_is_used(trained):
    _, ckpt = trained
    assert resolve_checkpoint(ckpt) == ckpt


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "nope.pt")


def test_default_search_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shipped = tmp_path / DEFAULT_CHECKPOINTS[1]
    shipped.parent.mkdir(parents=True)
    shipped.write_bytes(b"x")
    # With only the shipped checkpoint present it is chosen ...
    assert resolve_checkpoint(None).resolve() == shipped.resolve()

    trained_path = tmp_path / DEFAULT_CHECKPOINTS[0]
    trained_path.parent.mkdir(parents=True)
    trained_path.write_bytes(b"x")
    # ... but a model you trained yourself always wins.
    assert resolve_checkpoint(None).resolve() == trained_path.resolve()


def test_no_checkpoint_anywhere_explains_itself(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="quickstart"):
        resolve_checkpoint(None)


# --- export ----------------------------------------------------------------


def test_export_halves_the_file(trained, tmp_path):
    _, ckpt = trained
    info = export_checkpoint(ckpt, tmp_path / "small.pt")
    assert info["export_mb"] < info["source_mb"] * 0.6


def test_exported_checkpoint_is_numerically_equivalent(trained, tmp_path):
    """Half precision on disk, float32 in the model: predictions must match to
    the precision the cast allows, or the shipped model is not the one tested."""
    _, ckpt = trained
    export_checkpoint(ckpt, tmp_path / "small.pt")

    original, tok, _, _ = load_checkpoint(ckpt)
    exported, tok2, _, _ = load_checkpoint(tmp_path / "small.pt")
    original.eval(), exported.eval()

    x = torch.randint(0, original.cfg.vocab_size, (2, 32))
    with torch.no_grad():
        a, la, _ = original(x, x)
        b, lb, _ = exported(x, x)
    assert torch.allclose(a, b, atol=2e-2)
    assert abs(la.item() - lb.item()) < 2e-3
    assert tok2.merges == tok.merges


def test_exported_checkpoint_carries_its_tokenizer_and_config(trained, tmp_path):
    _, ckpt = trained
    export_checkpoint(ckpt, tmp_path / "small.pt")
    model, tok, cfg, raw = load_checkpoint(tmp_path / "small.pt")
    assert tok.vocab_size == model.cfg.vocab_size
    assert raw["fisher"] is not None
    assert raw["step"] > 0


def test_export_can_drop_fisher(trained, tmp_path):
    _, ckpt = trained
    with_f = export_checkpoint(ckpt, tmp_path / "a.pt", keep_fisher=True)
    without_f = export_checkpoint(ckpt, tmp_path / "b.pt", keep_fisher=False)
    assert without_f["export_mb"] < with_f["export_mb"]
    assert load_checkpoint(tmp_path / "b.pt")[3]["fisher"] is None


def test_learner_handles_a_half_precision_fisher(trained, tmp_path):
    """EWC must not silently become a half-precision computation."""
    from aria.config import LearnerConfig
    from aria.learner import OnlineLearner

    _, ckpt = trained
    export_checkpoint(ckpt, tmp_path / "small.pt")
    model, tok, _, raw = load_checkpoint(tmp_path / "small.pt")
    learner = OnlineLearner(model, tok, LearnerConfig(plasticity="full"),
                            fisher=raw["fisher"], state_dir=tmp_path / "online")
    assert learner.fisher
    for v in learner.fisher.values():
        assert v.dtype == torch.float32
    assert learner.observe(["hello", "hello there my friend"]) is not None


def test_export_command(trained, tmp_path, capsys):
    _, ckpt = trained
    out = tmp_path / "exported.pt"
    assert main(["export", "--checkpoint", str(ckpt), "--out", str(out)]) == 0
    assert out.exists()
    assert "MB ->" in capsys.readouterr().out


def test_chat_finds_the_shipped_checkpoint_without_flags(trained, tmp_path, monkeypatch):
    """The whole point of resolution: `aria chat` with no arguments works in a
    fresh clone that ships a checkpoint."""
    from aria.chat import ChatSession
    _, ckpt = trained
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / DEFAULT_CHECKPOINTS[1]
    dest.parent.mkdir(parents=True)
    export_checkpoint(ckpt, dest)

    session = ChatSession()
    assert isinstance(session.model, GPT)
    text, report = session.turn("hello")
    assert isinstance(text, str) and report is not None
