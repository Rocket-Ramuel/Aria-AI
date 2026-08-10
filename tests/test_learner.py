"""Tests for the continual-learning machinery.

These are the tests that matter most: the whole design exists to make online
learning safe, so each safeguard gets its own check.
"""

import random

import numpy as np
import pytest
import torch

from aria.config import LearnerConfig, ModelConfig
from aria.data import TokenStream, encode_dialogue
from aria.learner import OnlineLearner, resume_learned_weights
from aria.memory import Journal, ReplayBuffer
from aria.model import GPT, IGNORE_INDEX, LoRALinear
from aria.tokenizer import BPETokenizer

CORPUS = (
    "the cat sat on the mat and looked at the rain outside the window. "
    "she opened the door and stepped into the cold morning air. "
    "they walked along the river until the light began to fade. "
) * 60


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=520)


@pytest.fixture
def model(tok):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=4, n_kv_head=2,
                      n_embd=32, block_size=64)
    return GPT(cfg)


@pytest.fixture
def stream(tmp_path, tok):
    ids = np.array(tok.encode(CORPUS, allowed_special=False), dtype=np.uint16)
    p = tmp_path / "train.bin"
    ids.tofile(p)
    return TokenStream(p, 64)


def make_learner(model, tok, tmp_path, stream=None, **overrides):
    cfg = LearnerConfig(**{
        "plasticity": "lora", "lora_rank": 4, "learning_rate": 1e-3,
        "replay_batch": 2, "health_interval": 4, "consolidate_interval": 1000,
        "surprise_gate": False, **overrides,
    })
    return OnlineLearner(model, tok, cfg, pretrain_stream=stream,
                         state_dir=tmp_path / "online")


# --- encoding -------------------------------------------------------------


def test_only_aria_turns_carry_loss(tok):
    x, y = encode_dialogue(tok, ["hello there", "hello to you", "how are you"], 64)
    assert len(x) == len(y)
    supervised = [i for i, t in enumerate(y) if t != IGNORE_INDEX]
    assert supervised, "at least Aria's turn must be supervised"
    # Every supervised target must be reachable from an Aria segment: the
    # user's own words are never a training target.
    aria_span_started = False
    for i, tid in enumerate(x):
        if tid == tok.aria_id:
            aria_span_started = True
        elif tid == tok.user_id:
            aria_span_started = False
        if y[i] != IGNORE_INDEX:
            assert aria_span_started


def test_encode_dialogue_rejects_degenerate_input(tok):
    assert encode_dialogue(tok, ["hi"], 64) is None      # no Aria turn to learn from
    assert encode_dialogue(tok, [], 64) is None


# --- the basic learning loop ---------------------------------------------


def test_observing_lowers_loss_on_that_exchange(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path)
    turns = ["who are you", "I am Aria and I live in the terminal"]
    reports = [learner.observe(turns) for _ in range(6)]
    assert all(r.applied for r in reports)
    assert reports[-1].loss_after < reports[0].loss_before


def test_learning_persists_through_save_and_reload(model, tok, tmp_path, stream):
    learner = make_learner(model, tok, tmp_path, stream)
    turns = ["what is your name", "my name is Aria"]
    for _ in range(8):
        learner.observe(turns)
    learned_loss = learner.sequence_loss(encode_dialogue(tok, turns, 64))
    learner.save()

    fresh = GPT(model.cfg)
    assert resume_learned_weights(fresh, tmp_path / "online")
    l2 = make_learner(fresh, tok, tmp_path, stream)
    assert l2.sequence_loss(encode_dialogue(tok, turns, 64)) == pytest.approx(
        learned_loss, abs=1e-4
    )
    assert l2.updates_applied == learner.updates_applied


def test_base_weights_are_frozen_in_lora_mode(model, tok, tmp_path):
    before = {n: p.clone() for n, p in model.named_parameters()}
    learner = make_learner(model, tok, tmp_path)
    for _ in range(4):
        learner.observe(["hello", "hello back to you"])
    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        key = n.replace(".base.weight", ".weight")
        if key in before:
            assert torch.equal(before[key], p), f"{n} moved but should be frozen"


def test_full_plasticity_moves_real_weights(model, tok, tmp_path):
    before = {n: p.clone() for n, p in model.named_parameters()}
    learner = make_learner(model, tok, tmp_path, plasticity="full",
                           ewc_lambda=0.0, l2_anchor=0.0)
    for _ in range(4):
        learner.observe(["hello", "hello back to you"])
    moved = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert len(moved) > 5


# --- safeguards -----------------------------------------------------------


def test_surprise_gate_skips_the_unsurprising(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, surprise_gate=True,
                           surprise_floor=0.0, learning_rate=3e-3)
    turns = ["hello", "hello to you as well"]
    reports = [learner.observe(turns) for _ in range(25)]
    assert any(not r.applied for r in reports), "gate never fired"
    assert reports[0].applied or reports[1].applied
    # Skipped turns are still remembered for later rehearsal.
    assert len(learner.replay) == 25


def test_trust_region_caps_drift(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, learning_rate=0.5,
                           trust_radius=0.02, steps_per_turn=3, l2_anchor=0.0)
    for _ in range(12):
        r = learner.observe(["teach me", "the mitochondrion is the powerhouse"])
        assert r.drift <= 0.02 + 1e-5

    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                delta = (mod.lora_B @ mod.lora_A) * mod.scaling
                assert float(delta.norm() / mod.base.weight.norm()) <= 0.02 + 1e-5


def test_full_mode_trust_region_caps_drift(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, plasticity="full",
                           learning_rate=0.5, trust_radius=0.01,
                           ewc_lambda=0.0, l2_anchor=0.0)
    for _ in range(8):
        r = learner.observe(["teach me", "a fact worth remembering"])
        assert r.drift <= 0.01 + 1e-5


def test_ewc_penalises_movement_away_from_the_anchor(model, tok, tmp_path):
    fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}

    def drift_with(lam):
        torch.manual_seed(0)
        m = GPT(model.cfg)
        cfg = LearnerConfig(plasticity="full", learning_rate=5e-3, replay_batch=1,
                            ewc_lambda=lam, l2_anchor=0.0, trust_radius=10.0,
                            surprise_gate=False, health_interval=10_000,
                            consolidate_interval=10_000)
        lr = OnlineLearner(m, tok, cfg, fisher=fisher,
                           state_dir=tmp_path / f"ewc{lam}")
        for _ in range(6):
            r = lr.observe(["hello", "a completely novel sentence about otters"])
        return r.drift

    assert drift_with(1e4) < drift_with(0.0)


def test_rollback_restores_weights_when_the_canary_degrades(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, health_interval=1,
                           health_patience=1, learning_rate=0.2,
                           trust_radius=10.0, l2_anchor=0.0, steps_per_turn=4)
    # Force the canary bar impossibly low so the next health check must fail.
    learner.canary_baseline = 1e-6
    good = {n: p.clone() for n, p in model.named_parameters() if "lora_" in n}
    lr_before = learner.lr

    r = learner.observe(["nonsense", "zzz qqq zzz qqq zzz qqq"])
    assert r.rolled_back
    assert learner.lr < lr_before
    for n, p in model.named_parameters():
        if "lora_" in n:
            assert torch.equal(good[n], p), "rollback did not restore the weights"


def test_health_patience_delays_rollback(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, health_interval=1,
                           health_patience=3, learning_rate=0.2,
                           trust_radius=10.0, l2_anchor=0.0)
    learner.canary_baseline = 1e-6
    rolled = [learner.observe(["x", "y z w"]).rolled_back for _ in range(3)]
    assert rolled == [False, False, True]


def test_consolidation_merges_adapters_into_real_weights(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, consolidate_interval=1000)
    for _ in range(5):
        learner.observe(["who are you", "I am Aria"])

    ex = encode_dialogue(tok, ["who are you", "I am Aria"], 64)
    before = learner.sequence_loss(ex)
    base_before = {n: p.clone() for n, p in model.named_parameters()
                   if n.endswith("q_proj.base.weight")}
    assert base_before

    learner.canary_baseline = 1e9      # guarantee the health gate passes
    learner.consolidate()

    # The function of the model is preserved ...
    assert learner.sequence_loss(ex) == pytest.approx(before, abs=1e-4)
    # ... but it now lives in the ordinary weight matrices (which are wrapped
    # again by fresh adapters, so the name is unchanged) ...
    named = dict(model.named_parameters())
    for n, old in base_before.items():
        assert not torch.equal(old, named[n])
    # ... and the adapters have restarted from zero.
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            assert torch.count_nonzero(mod.lora_B) == 0


def test_consolidation_is_skipped_when_unhealthy(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path)
    for _ in range(3):
        learner.observe(["hi", "hello there"])
    learner.canary_baseline = 1e-9     # nothing can pass this
    snapshot = {n: p.clone() for n, p in model.named_parameters()}
    learner.consolidate()
    for n, p in model.named_parameters():
        assert torch.equal(snapshot[n], p)


def test_replay_mixes_conversation_and_corpus(model, tok, tmp_path, stream):
    learner = make_learner(model, tok, tmp_path, stream, replay_batch=6,
                           pretrain_replay_frac=0.5)
    for i in range(5):
        learner.observe([f"question {i}", f"answer number {i}"])
    r = learner.observe(["another question", "another answer"])
    assert r.replayed == 6


def test_replay_still_works_without_a_corpus(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, stream=None, replay_batch=4)
    for i in range(6):
        learner.observe([f"q{i}", f"a{i} with some words"])
    assert learner.observe(["q", "a with some words"]).replayed > 0


def test_rehearsal_protects_an_old_lesson(model, tok, tmp_path, stream):
    """The headline claim: learning new things must not erase old ones."""
    learner = make_learner(model, tok, tmp_path, stream, learning_rate=2e-3,
                           replay_batch=6, consolidate_interval=10_000)
    old = ["what is your name", "my name is Aria and I remember you"]
    for _ in range(15):
        learner.observe(old, weight=2.0)
    ex_old = encode_dialogue(tok, old, 64)
    after_learning_old = learner.sequence_loss(ex_old)

    for i in range(40):
        learner.observe([f"tell me about topic {i}",
                         f"topic {i} concerns rivers and the cold morning air"])

    after_learning_new = learner.sequence_loss(ex_old)
    # Some regression is expected; wholesale forgetting is not.
    assert after_learning_new < after_learning_old * 1.6


def test_canary_is_never_trained_on(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path)
    canary_texts = {tok.decode(x, skip_special=True) for x in learner.canaries[0].tolist()}
    for _ in range(3):
        learner.observe(["hello", "hi"])
    remembered = {t for item in learner.replay.items for t in item["turns"]}
    assert not (remembered & canary_texts)


# --- memory ---------------------------------------------------------------


def test_replay_buffer_respects_capacity_and_persists(tmp_path):
    buf = ReplayBuffer(capacity=10, seed=1)
    for i in range(200):
        buf.add([f"u{i}", f"a{i}"])
    assert len(buf) == 10
    assert buf.seen == 200

    buf.save(tmp_path / "r.json")
    again = ReplayBuffer.load(tmp_path / "r.json")
    assert len(again) == 10 and again.seen == 200
    assert again.items[0]["turns"] == buf.items[0]["turns"]


def test_replay_buffer_ignores_incomplete_exchanges():
    buf = ReplayBuffer(capacity=4)
    assert not buf.add(["lonely turn"])
    assert len(buf) == 0


def test_replay_sampling_favours_weighted_items():
    buf = ReplayBuffer(capacity=100, seed=2)
    buf.add(["a", "b"], weight=100.0, kind="correction")
    for i in range(20):
        buf.add([f"u{i}", f"a{i}"], weight=0.01)
    picks = buf.sample(200)
    corrections = sum(1 for p in picks if p["kind"] == "correction")
    assert corrections > len(picks) * 0.5


def test_journal_summarises(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.write(event="update", applied=True, loss_after=1.0)
    j.write(event="update", applied=False)
    j.write(event="rollback")
    s = j.summary()
    assert s == {"turns_seen": 2, "updates_applied": 1, "updates_skipped": 1,
                 "rollbacks": 1, "consolidations": 0, "mean_loss_after_update": 1.0}


def test_status_reports_trainable_subset(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path)
    st = learner.status()
    assert st["plasticity"] == "lora"
    assert 0 < st["trainable_params"] < model.num_params()


def test_unknown_plasticity_mode_rejected(model, tok, tmp_path):
    with pytest.raises(ValueError):
        OnlineLearner(model, tok, LearnerConfig(plasticity="nope"),
                      state_dir=tmp_path / "x")
