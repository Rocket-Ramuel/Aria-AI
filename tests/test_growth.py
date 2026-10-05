"""Growing a model, and keeping a large one manageable."""

import pytest
import torch

from aria.chat import ChatSession
from aria.config import LearnerConfig, ModelConfig
from aria.data import encode_dialogue
from aria.learner import OnlineLearner, bf16_supported
from aria.model import GPT, LoRALinear, grow
from aria.optim import LowMemoryAdam
from aria.pretrain import create_blank_checkpoint, export_checkpoint, load_checkpoint
from aria.storage import dequantize_state_dict, int8_state_dict
from aria.tokenizer import BPETokenizer

TEXT = ("The kettle was on before anyone woke. Rain fell on the slate roof all night. "
        "She folded the letter twice and put it in her coat. ") * 30


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(TEXT, vocab_size=400)


def tiny(tok, layers=2):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=tok.vocab_size, n_layer=layers, n_head=4,
                           n_kv_head=2, n_embd=32, block_size=64))


# --- growth -----------------------------------------------------------------


def test_growing_changes_nothing_until_she_learns(tok):
    model = tiny(tok).eval()
    x = torch.randint(0, tok.vocab_size, (2, 30))
    before = model(x, x)[0]
    assert grow(model, 3) == 5
    assert torch.equal(model(x, x)[0], before)        # bit for bit


def test_grown_layers_learn_even_in_lora_mode(tok, tmp_path):
    model = tiny(tok)
    grow(model, 1)
    learner = OnlineLearner(model, tok, LearnerConfig(surprise_gate=False,
                                                      learning_rate=1e-2),
                            state_dir=tmp_path / "o", grown_from=2)
    # Old blocks get adapters; the grown block is trained directly.
    assert isinstance(model.blocks[0].attn.q_proj, LoRALinear)
    assert not isinstance(model.blocks[2].attn.q_proj, LoRALinear)
    o = model.blocks[2].attn.o_proj.weight
    assert o.requires_grad and torch.count_nonzero(o) == 0
    for _ in range(3):
        learner.observe(["hello there", "the kettle was on"])
    assert torch.count_nonzero(o) > 0, "the grown layer never started learning"


def test_a_session_grows_and_remembers_it(tok, tmp_path):
    ckpt = create_blank_checkpoint(tmp_path / "b" / "base.pt", size="tiny", block_size=64)
    s = ChatSession(checkpoint=ckpt, max_new_tokens=4)
    s.upload("t.txt", TEXT.encode(), passes=1)
    probe = s.learner._document_windows(TEXT)[:2]
    learned = s.learner.mean_loss(probe)
    msg = s.grow(2)
    assert "from 4 to 6 layers" in msg
    assert s.learner.mean_loss(probe) == pytest.approx(learned, abs=1e-4)

    again = ChatSession(checkpoint=ckpt)                # a later session
    assert again.model.cfg.n_layer == 6
    assert again.learner.mean_loss(probe) == pytest.approx(learned, abs=2e-3)
    assert again.learner.grown_from == 4

    out = tmp_path / "mine.pt"
    export_checkpoint(ckpt, out, learned=again.state_dir / "learned.pt")
    assert load_checkpoint(out)[2].model.n_layer == 6


def test_she_grows_when_she_has_read_more_than_she_has_room_for(tok, tmp_path):
    ckpt = create_blank_checkpoint(tmp_path / "b" / "base.pt", size="tiny", block_size=64)
    s = ChatSession(checkpoint=ckpt, learner_overrides={"tokens_per_param": 1e-4,
                                                         "grow_max_factor": 1.5})
    _, summary = s.upload("t.txt", TEXT.encode(), passes=1)
    assert "room for, so she grew from 4 to 5 layers" in summary
    s.upload("t.txt", TEXT.encode(), passes=1)
    _, summary = s.upload("t.txt", TEXT.encode(), passes=1)
    assert s.model.cfg.n_layer == 6                      # the 1.5x ceiling
    assert "grew" not in summary


def test_room_left_counts_pretraining(tok, tmp_path):
    learner = OnlineLearner(tiny(tok), tok, LearnerConfig(), state_dir=tmp_path / "o",
                            prior_tokens=10**12)
    assert learner.room_left() == 0.0
    fresh = OnlineLearner(tiny(tok), tok, LearnerConfig(), state_dir=tmp_path / "p")
    assert fresh.room_left() == 1.0


def test_no_growing_in_the_middle_of_an_upload(tok, tmp_path):
    ckpt = create_blank_checkpoint(tmp_path / "b" / "base.pt", size="tiny", block_size=64)
    s = ChatSession(checkpoint=ckpt)
    gen = s.iter_learn_file(TEXT.encode(), "t.txt", passes=1)
    next(gen)
    with pytest.raises(RuntimeError, match="reading a document"):
        s.grow(1)
    gen.close()


# --- memory -------------------------------------------------------------------


def test_low_memory_adam_learns_like_adam_with_a_quarter_of_the_state(tok):
    def train(opt_cls):
        model = tiny(tok)
        opt = opt_cls(list(model.parameters()), lr=3e-3)
        ex = encode_dialogue(tok, ["what was on", "the kettle was on before anyone woke"], 64)
        x, y = torch.tensor([ex[0]]), torch.tensor([ex[1]])
        for _ in range(60):
            opt.zero_grad()
            model(x, y)[1].backward()
            opt.step()
        with torch.no_grad():
            return float(model(x, y)[1]), opt

    adam_loss, _ = train(torch.optim.AdamW)
    low_loss, low = train(LowMemoryAdam)
    assert low_loss < 0.5 and low_loss < adam_loss * 3
    n_params = sum(p.numel() for p in tiny(tok).parameters())
    assert low.state_bytes() < n_params * 2.6          # vs 8 bytes for AdamW


def test_checkpointing_gives_the_same_gradients(tok):
    x = torch.randint(0, tok.vocab_size, (2, 40))
    grads = []
    for ckpt in (False, True):
        model = tiny(tok).train()
        model.checkpointing = ckpt
        model(x, x)[1].backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-6)


def test_memory_saver_switches_on_for_large_models(tok, tmp_path):
    small = OnlineLearner(tiny(tok), tok, LearnerConfig(), state_dir=tmp_path / "a")
    assert not small.saving_memory and isinstance(small.opt, torch.optim.AdamW)
    forced = OnlineLearner(tiny(tok), tok, LearnerConfig(memory_saver="on",
                                                         surprise_gate=False),
                           state_dir=tmp_path / "b")
    assert forced.saving_memory and isinstance(forced.opt, LowMemoryAdam)
    assert forced.model.checkpointing
    assert all(v.dtype == torch.float16 for v in forced.last_good.values())
    r = forced.observe(["hello there", "the kettle was on"])
    assert r.applied and r.loss_after < r.loss_before
    with pytest.raises(ValueError):
        OnlineLearner(tiny(tok), tok, LearnerConfig(memory_saver="maybe"),
                      state_dir=tmp_path / "c")


@pytest.mark.skipif(not bf16_supported("cpu"), reason="no native bfloat16 on this CPU")
def test_bfloat16_arithmetic_learns_and_talks(tmp_path):
    ckpt = create_blank_checkpoint(tmp_path / "b" / "base.pt", size="tiny", block_size=64)
    s = ChatSession(checkpoint=ckpt, max_new_tokens=8,
                    learner_overrides={"memory_saver": "on"})
    assert s.learner.fast_math
    report, _ = s.upload("t.txt", TEXT.encode(), passes=2)
    assert report.loss_after < report.loss_before
    assert isinstance(s.reply("hello"), str)
    assert all(p.dtype == torch.float32 for p in s.model.parameters())


def test_int8_weights_are_a_quarter_of_float32_and_nearly_lossless(tok, tmp_path):
    torch.manual_seed(0)
    # Wide enough that every matrix is quantised (small ones stay float16).
    model = GPT(ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=4,
                            n_kv_head=2, n_embd=128, block_size=64))
    q = int8_state_dict(model.state_dict())
    assert q["tok_emb.weight"] is q["lm_head.weight"]    # tied, stored once
    back = dequantize_state_dict(q)
    for k, v in model.state_dict().items():
        err = (back[k].float() - v).abs().max() / v.abs().max().clamp_min(1e-12)
        assert err < 0.01, k
    torch.save(q, tmp_path / "q.pt")
    torch.save(model.state_dict(), tmp_path / "f.pt")
    assert (tmp_path / "q.pt").stat().st_size < (tmp_path / "f.pt").stat().st_size * 0.3
