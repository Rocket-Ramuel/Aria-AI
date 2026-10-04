"""Tests for learning someone's voice: uploads, style mirroring, blank models."""

import numpy as np
import pytest
import torch

from aria.chat import ChatSession
from aria.config import LearnerConfig, ModelConfig, blank_learner_config
from aria.data import encode_dialogue
from aria.learner import OnlineLearner
from aria.model import GPT
from aria.pretrain import create_blank_checkpoint, load_checkpoint
from aria.tokenizer import BPETokenizer

CORPUS = (
    "the cat sat on the mat and looked at the rain outside the window. "
    "she opened the door and stepped into the cold morning air. "
) * 60

# A distinctive voice, nothing like the corpus above.
VOICE = (
    "Right then, love, kettle's on. Proper grim out there today, innit. "
    "Reckon we'll have a brew and a natter before the footy starts. "
    "Mind you, our Gary says the bus were late again, the daft thing. "
) * 6


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS + VOICE, vocab_size=600)


@pytest.fixture
def model(tok):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=4,
                           n_kv_head=2, n_embd=32, block_size=64))


def make_learner(model, tok, tmp_path, **overrides):
    cfg = LearnerConfig(**{
        "plasticity": "lora", "lora_rank": 4, "learning_rate": 2e-3,
        "replay_batch": 2, "health_interval": 4, "consolidate_interval": 1000,
        "surprise_gate": False, **overrides,
    })
    return OnlineLearner(model, tok, cfg, state_dir=tmp_path / "online")


def voice_loss(learner):
    return learner.mean_loss(learner._document_windows(VOICE)[:4])


# --- documents ------------------------------------------------------------


def test_learning_a_document_lowers_loss_on_it(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, plasticity="full", trust_radius=0.5,
                           document_batch=2)
    before = voice_loss(learner)
    report = learner.learn_document(VOICE, passes=8)
    assert report.applied and report.steps > 1
    assert report.loss_after < report.loss_before
    assert voice_loss(learner) < before * 0.9


def test_a_document_covers_all_of_the_text(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path)
    windows = learner._document_windows(VOICE)
    assert len(windows) > 1
    covered = set()
    for x, _ in windows:
        covered.update(t for t in x if t >= tok.n_special)
    # Documents are encoded sentence by sentence, the way chat turns are.
    from aria.documents import iter_units
    expected = {t for u in iter_units(VOICE.splitlines())
                for t in tok.encode(" " + u) if t >= tok.n_special}
    assert covered == expected
    for x, y in windows:
        assert len(x) <= model.cfg.block_size
        assert x[:2] == [tok.bos_id, tok.aria_id]


def test_document_is_filed_for_rehearsal(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path)
    learner.learn_document("\n\n".join([VOICE[:200]] * 6), passes=1)
    kinds = {item["kind"] for item in learner.replay.items}
    assert kinds == {"document"}
    assert learner._replay_examples(4), "documents must be rehearsable"


def test_document_steps_cannot_skip_a_consolidation(model, tok, tmp_path):
    """Consolidation used to fire on `updates_applied % interval == 0` inside
    `observe` only, so a document upload could step right over it."""
    learner = make_learner(model, tok, tmp_path, consolidate_interval=3,
                           document_batch=1, health_check=False)
    learner.observe(["hello", "hello to you"])            # update 1
    report = learner.learn_document(VOICE, passes=1)      # several more
    assert report.consolidated
    assert any(r.get("event") == "consolidate" and not r.get("skipped")
               for r in learner.journal.read())


def test_document_learning_stops_after_repeated_rollbacks(model, tok, tmp_path):
    """One rollback slows an upload down; three in a row mean the material
    itself is hurting her English, and the upload stops."""
    from aria.learner import MAX_ROLLBACKS_PER_UPLOAD
    learner = make_learner(model, tok, tmp_path, health_interval=1,
                           health_patience=1, document_batch=1)
    learner.canary_baseline = 1e-6                        # every check fails
    lr_before = learner.lr
    report = learner.learn_document(VOICE, passes=5)
    assert report.rolled_back
    assert report.steps == MAX_ROLLBACKS_PER_UPLOAD
    assert learner.lr < lr_before


def test_learning_a_transcript_teaches_that_persons_replies(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, trust_radius=0.5)
    convos = [["are you coming tonight", "Reckon I'll stop in, love, proper grim out"]] * 4
    ex = encode_dialogue(tok, convos[0], 64)
    before = learner.sequence_loss(ex)
    report = learner.learn_dialogues(convos, passes=4)
    assert report.applied
    assert learner.sequence_loss(ex) < before
    assert {i["kind"] for i in learner.replay.items} == {"transcript"}


# --- style mirroring --------------------------------------------------------


def test_style_mirror_learns_the_users_own_words(model, tok, tmp_path):
    said = "Proper grim out there today, innit, kettle's on"

    def user_loss(mirror):
        torch.manual_seed(0)
        m = GPT(model.cfg)
        learner = make_learner(m, tok, tmp_path / str(mirror), style_mirror=mirror,
                               style_weight=1.0, trust_radius=0.5)
        probe = learner._text_example(said)
        for _ in range(6):
            learner.observe([said, "the cat sat on the mat"])
        return learner.sequence_loss(probe)

    assert user_loss(True) < user_loss(False)


def test_own_replies_can_be_excluded(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, learn_own_replies=False,
                           plasticity="full", ewc_lambda=0.0, l2_anchor=0.0,
                           trust_radius=0.0)
    babble = ["Reckon we'll have a brew and a natter", "zq zq zq zq zq zq zq"]
    reply = encode_dialogue(tok, babble, 64)
    before = learner.sequence_loss(reply)
    for _ in range(5):
        learner.observe(babble)
    # Aria's own babble was not reinforced; the user's line was learned.
    assert learner.sequence_loss(reply) >= before - 0.05
    assert learner.sequence_loss(learner._text_example(babble[0])) < before


def test_corrections_are_learned_even_without_own_replies(model, tok, tmp_path):
    learner = make_learner(model, tok, tmp_path, learn_own_replies=False,
                           style_mirror=False, trust_radius=0.5)
    turns = ["what's the weather", "proper grim out there today"]
    assert not learner.observe(turns).applied
    assert learner.observe(turns, kind="correction", force=True).applied


def test_zero_trust_radius_disables_the_projection(model, tok, tmp_path):
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    learner = make_learner(model, tok, tmp_path, plasticity="full", trust_radius=0.0,
                           learning_rate=0.05, ewc_lambda=0.0, l2_anchor=0.0)
    for _ in range(5):
        r = learner.observe(["teach me", "a fact worth remembering"])
    assert r.drift is None                     # nothing anchors the weights
    moved = max(float((p.detach() - before[n]).norm() / before[n].norm())
                for n, p in model.named_parameters())
    assert moved > 0.05


# --- a model with no pretraining -------------------------------------------


@pytest.fixture
def blank_ckpt(tmp_path):
    return create_blank_checkpoint(tmp_path / "blank" / "base.pt", size="tiny",
                                   block_size=128)


def test_blank_checkpoint_knows_nothing(blank_ckpt):
    model, tok, cfg, ckpt = load_checkpoint(blank_ckpt)
    assert tok.merges == [] and tok.vocab_size == tok.n_special + 256
    assert ckpt["fisher"] is None and ckpt["step"] == 0
    assert cfg.learner == blank_learner_config()
    # Byte-level: anything at all is encodable, nothing is assumed.
    text = "Ça va? 日本, innit"
    assert tok.decode(tok.encode(text)) == text


def test_blank_session_uses_blank_learner_settings(blank_ckpt, tmp_path):
    s = ChatSession(checkpoint=blank_ckpt, data_dir=tmp_path / "nodata")
    assert s.state_dir == blank_ckpt.parent / "online"
    assert s.learner.cfg.plasticity == "full"
    assert not s.learner.cfg.health_check
    assert s.max_new_tokens == 240           # byte-level replies need more tokens
    # Command-line overrides change one setting, not all of them.
    s2 = ChatSession(checkpoint=blank_ckpt, state_dir=tmp_path / "o2",
                     learner_overrides={"learning_rate": 5e-4})
    assert s2.learner.cfg.learning_rate == 5e-4
    assert s2.learner.cfg.plasticity == "full"


def test_blank_model_learns_a_voice_from_an_upload(blank_ckpt, tmp_path):
    s = ChatSession(checkpoint=blank_ckpt, max_new_tokens=8)
    probe = s.learner._document_windows(VOICE)[:2]
    before = s.learner.mean_loss(probe)
    report, summary = s.upload("gran.txt", VOICE.encode(), passes=3)
    assert "learned gran.txt" in summary
    after = s.learner.mean_loss(probe)
    assert after < before * 0.8, (before, after)
    assert (s.state_dir / "learned.pt").exists()   # uploads are saved at once


def test_upload_of_a_transcript_needs_a_known_speaker(blank_ckpt, tmp_path):
    s = ChatSession(checkpoint=blank_ckpt)
    log = b"Sam: hiya\nJo: alright love\nSam: you well\nJo: can't complain\n"
    with pytest.raises(ValueError, match="speakers are"):
        s.upload("chat.txt", log, speaker="Gary")
    _, summary = s.upload("chat.txt", log, speaker="jo", passes=1)
    assert "replies by jo" in summary
    _, summary = s.upload("chat.txt", log, passes=1)
    assert "a transcript" in summary


def test_session_ignores_a_corpus_from_another_tokenizer(tmp_path, blank_ckpt):
    """Rehearsing a corpus encoded with a different vocabulary used to feed
    out-of-range token ids into the model."""
    data = tmp_path / "data"
    data.mkdir()
    other = BPETokenizer.train(CORPUS, vocab_size=400)
    other.save(data / "tokenizer.json")
    np.array(other.encode(CORPUS), dtype=np.uint16).tofile(data / "train.bin")
    s = ChatSession(checkpoint=blank_ckpt, data_dir=data,
                    learner_overrides={"pretrain_replay_frac": 0.5})
    assert s.learner.pretrain_stream is None


def test_documents_are_also_learned_as_replies(model, tok, tmp_path):
    """A blank model trained only on continuation never sees the
    <user> ... <aria> context replies are generated in."""
    learner = make_learner(model, tok, tmp_path)
    text = "Right then, love. Kettle's on! Proper grim out there today, innit."
    pairs = learner._document_exchanges(text)
    assert len(pairs) == 2
    x, y = pairs[0]
    assert tok.user_id in x and tok.aria_id in x
    assert "Kettle's on!" in tok.decode([t for t in y if t >= 0], skip_special=True)
    assert "Right then" not in tok.decode([t for t in y if t >= 0], skip_special=True)
