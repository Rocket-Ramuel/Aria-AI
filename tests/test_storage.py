"""How compactly Aria is kept on disk, and that nothing grows without bound."""

import json

import pytest
import torch

from aria.config import LearnerConfig, ModelConfig, blank_learner_config
from aria.learner import OnlineLearner, resume_learned_weights
from aria.memory import MAX_STORED_TURN_CHARS, Journal, ReplayBuffer
from aria.model import GPT
from aria.storage import decode_fisher, encode_fisher, half_state_dict
from aria.tokenizer import BPETokenizer

TEXT = "the cat sat on the mat and looked at the rain outside the window. " * 40


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(TEXT, vocab_size=400)


def tiny(tok):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=4, n_kv_head=2,
                           n_embd=32, block_size=64))


def unique_params(model):
    return sum(p.numel() for p in model.parameters())     # tied weights counted once


# --- weights --------------------------------------------------------------


def test_half_state_dict_stores_tied_weights_once(tok, tmp_path):
    model = tiny(tok)
    sd = half_state_dict(model.state_dict())
    assert sd["tok_emb.weight"] is sd["lm_head.weight"]
    assert all(v.dtype == torch.float16 for v in sd.values() if v.is_floating_point())
    torch.save(sd, tmp_path / "w.pt")
    size = (tmp_path / "w.pt").stat().st_size
    assert size < unique_params(model) * 2 * 1.1         # ~2 bytes per weight


def test_values_too_big_for_half_precision_stay_full_precision():
    sd = half_state_dict({"ok": torch.ones(3), "huge": torch.tensor([1e6, 1.0])})
    assert sd["ok"].dtype == torch.float16
    assert sd["huge"].dtype == torch.float32 and sd["huge"][0] == 1e6


def test_learned_weights_are_saved_compactly_and_reload_faithfully(tok, tmp_path):
    from aria.data import encode_dialogue
    model = tiny(tok)
    learner = OnlineLearner(model, tok, LearnerConfig(surprise_gate=False),
                            state_dir=tmp_path / "online")
    turns = ["what is it", "the cat sat on the mat"]
    for _ in range(4):
        learner.observe(turns)
    learner.save()

    path = tmp_path / "online" / "learned.pt"
    state = torch.load(path, weights_only=True)["model"]
    assert {v.dtype for v in state.values() if v.is_floating_point()} == {torch.float16}
    assert not any("lora_" in k for k in state)
    assert path.stat().st_size < unique_params(model) * 2 * 1.15
    assert not list(path.parent.glob("*.tmp"))            # atomic write left nothing

    fresh = tiny(tok)
    resume_learned_weights(fresh, tmp_path / "online")
    ex = encode_dialogue(tok, turns, 64)
    again = OnlineLearner(fresh, tok, LearnerConfig(), state_dir=tmp_path / "again")
    assert again.sequence_loss(ex) == pytest.approx(learner.sequence_loss(ex), abs=2e-3)


# --- Fisher -----------------------------------------------------------------


def test_fisher_takes_one_byte_per_weight_and_keeps_tiny_values():
    """float16 rounds Fisher values around 1e-7 to zero or to a few bits; the
    log encoding keeps every one of them to within a few percent."""
    torch.manual_seed(0)
    f = {"a": torch.rand(1000).pow(6) * 1e-3, "b": torch.zeros(10)}
    f["a"][:5] = torch.tensor([1e-12, 3e-10, 1e-8, 2e-7, 5e-7])
    enc = encode_fisher(f)
    assert enc["q"]["a"].dtype == torch.uint8
    dec = decode_fisher(enc)
    assert torch.equal(dec["b"], torch.zeros(10))
    rel = (dec["a"] - f["a"]).abs() / f["a"]
    assert float(rel.max()) < 0.15
    assert (dec["a"][:5] > 0).all()
    assert (f["a"][:5].half().float() == 0).any()        # what float16 would have done


def test_decode_accepts_every_stored_form():
    f = {"w": torch.tensor([1e-4, 2e-4])}
    assert decode_fisher(None) is None
    assert decode_fisher({k: v.half() for k, v in f.items()})["w"].dtype == torch.float32
    assert torch.allclose(decode_fisher(f)["w"], f["w"])


# --- memory -----------------------------------------------------------------


def test_a_blank_model_keeps_no_safety_copies(tok, tmp_path):
    """Anchor and rollback snapshots are each a full copy of the network.
    A blank model uses neither, so it doesn't pay for them."""
    learner = OnlineLearner(tiny(tok), tok, blank_learner_config(),
                            state_dir=tmp_path / "online")
    assert learner.anchor == {} and learner.last_good == {}
    report = learner.observe(["hello there you", "a reply"])
    assert report.applied and report.drift is None


def test_a_rollback_does_not_slow_learning_forever(tok, tmp_path):
    learner = OnlineLearner(tiny(tok), tok, LearnerConfig(surprise_gate=False),
                            state_dir=tmp_path / "online")
    configured = learner.lr
    learner.rollback(canary=99.0)
    assert learner.lr < configured
    learner.canary_baseline = 1e9                       # healthy
    for _ in range(5):
        learner.consolidate()
    assert learner.lr == configured


def test_journal_is_bounded_but_keeps_lifetime_totals(tmp_path):
    j = Journal(tmp_path / "journal.jsonl", max_bytes=2_000)
    for i in range(400):
        j.write(event="update", applied=i % 2 == 0, loss_after=1.0,
                turns=["some words " * 5, "a reply"])
    j.write(event="rollback")
    total = sum(p.stat().st_size for p in tmp_path.iterdir())
    assert total < 2 * 2_000 + 1_000, f"journal grew to {total} bytes"
    s = j.summary()
    assert s["updates_applied"] == 200 and s["updates_skipped"] == 200
    assert s["turns_seen"] == 400 and s["rollbacks"] == 1
    assert s["mean_loss_after_update"] == pytest.approx(1.0)


def test_a_huge_message_cannot_bloat_the_replay_buffer(tmp_path):
    buf = ReplayBuffer(capacity=10)
    buf.add(["x" * 1_000_000, "reply"])
    assert len(buf.items[0]["turns"][0]) == MAX_STORED_TURN_CHARS
    buf.save(tmp_path / "r.json")
    assert (tmp_path / "r.json").stat().st_size < 10_000


# --- export -------------------------------------------------------------------


def test_export_can_bake_in_what_was_learned(tmp_path):
    from aria.chat import ChatSession
    from aria.cli import main
    from aria.pretrain import create_blank_checkpoint, load_checkpoint

    ckpt = create_blank_checkpoint(tmp_path / "b" / "base.pt", size="tiny", block_size=64)
    s = ChatSession(checkpoint=ckpt, max_new_tokens=4)
    s.upload("t.txt", TEXT.encode(), passes=1)
    probe = s.learner._document_windows(TEXT)[:2]
    learned_loss = s.learner.mean_loss(probe)

    out = tmp_path / "mine.pt"
    assert main(["export", "--checkpoint", str(ckpt), "--with-learning",
                 "--out", str(out)]) == 0
    model, tok, cfg, _ = load_checkpoint(out)
    fresh = ChatSession(checkpoint=out, state_dir=tmp_path / "fresh")
    assert fresh.learner.mean_loss(probe) == pytest.approx(learned_loss, abs=5e-3)
    assert json.loads(json.dumps(cfg.to_dict()))["learner"]["plasticity"] == "full"
