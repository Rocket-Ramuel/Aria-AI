"""The brain-like parts: hippocampus (one-shot memory) and cortical areas."""

import pytest
import torch
import torch.nn.functional as F

from aria.chat import ChatSession, brain_map
from aria.config import LearnerConfig, ModelConfig
from aria.data import encode_dialogue
from aria.hippocampus import Hippocampus
from aria.learner import OnlineLearner
from aria.model import GPT, Areas, grow
from aria.pretrain import create_blank_checkpoint
from aria.sample import generate
from aria.tokenizer import BPETokenizer

TEXT = ("my dog is called Biscuit. my cat is called Pepper. I live in Leeds. "
        "the weather was cold and the river was high. ") * 40


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(TEXT, vocab_size=420)


@pytest.fixture
def model(tok):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=4,
                           n_kv_head=2, n_embd=32, block_size=64)).eval()


def answer_logprob(model, tok, memory, question, answer, word):
    x, y = encode_dialogue(tok, [question, answer], 64)
    out = model(torch.tensor([x]), torch.tensor([x]), return_hidden=True)
    if memory is not None:
        memory.focus(tok.encode(" " + question))
        lp = memory.recall(out[3][0], out[0][0])
    else:
        lp = F.log_softmax(out[0][0], -1)
    first = tok.encode(" " + word)[0]
    pos = next(i for i in range(len(y)) if y[i] == first)
    return float(lp[pos, first])


# --- hippocampus ----------------------------------------------------------------


def test_one_hearing_makes_the_fact_more_likely(model, tok):
    h = Hippocampus(model, tok, threshold=0.0)
    h.store(["my dog is called Biscuit", "a fine name"])
    h.store(["the weather was cold", "and the river was high"])
    with torch.no_grad():
        without = answer_logprob(model, tok, None, "what is my dog called?",
                                 "your dog is called Biscuit", "Biscuit")
        with_memory = answer_logprob(model, tok, h, "what is my dog called?",
                                     "your dog is called Biscuit", "Biscuit")
    assert with_memory > without + 1.0


def test_recall_focuses_on_the_episode_the_conversation_is_about(model, tok):
    h = Hippocampus(model, tok)
    h.store(["my dog is called Biscuit", "a fine name"])
    h.store(["the weather was cold", "and the river was high"])
    h.store(["I live in Leeds", "a northern city"])
    found = h.focus(tok.encode(" what is my dog called?"))
    assert found and "Biscuit" in found[0][1]
    assert h.focus(tok.encode(" was it cold?")) == [] or \
        "Biscuit" not in h.focus(tok.encode(" was it cold?"))[0][1]


def test_with_nothing_on_topic_the_cortex_speaks_alone(model, tok):
    h = Hippocampus(model, tok)
    h.store(["my dog is called Biscuit", "a fine name"])
    h.focus(tok.encode(" the river was high"))     # no shared rare words
    x = torch.randint(0, tok.vocab_size, (12,))
    with torch.no_grad():
        out = model(x[None], x[None], return_hidden=True)
    assert torch.allclose(h.recall(out[3][0], out[0][0]), F.log_softmax(out[0][0], -1))


def test_memory_is_capped(model, tok):
    h = Hippocampus(model, tok, capacity_tokens=100)
    for i in range(20):
        h.store([f"message number {i} about the river", "and a reply"])
    assert len(h) <= 100
    assert len(h.episodes) < 20 and "19" in list(h.episodes.values())[-1]
    assert set(h.owner.unique().tolist()) == set(h.episodes)


def test_the_learner_remembers_and_sleep_reencodes(model, tok, tmp_path):
    learner = OnlineLearner(model, tok, LearnerConfig(surprise_gate=False,
                                                      consolidate_interval=1000),
                            state_dir=tmp_path / "o")
    learner.observe(["my dog is called Biscuit", "a fine name"])
    learner.observe(["I live in Leeds", "a northern city"])
    h = learner.hippocampus
    assert len(h.episodes) == 2
    before = h.keys.clone()
    for _ in range(3):
        learner.observe(["the weather was cold", "and the river was high"])
    learner.canary_baseline = 1e9
    learner.consolidate()                          # sleep
    assert len(h.episodes) == 5
    assert not torch.equal(h.keys[: before.shape[0]], before)   # re-encoded
    learner.save()
    torch.manual_seed(0)
    fresh = GPT(model.cfg)
    again = OnlineLearner(fresh, tok, LearnerConfig(), state_dir=tmp_path / "o")
    assert len(again.hippocampus.episodes) == 5     # rebuilt from replay text


def test_episodes_are_single_exchanges_not_the_context_window(model, tok, tmp_path):
    learner = OnlineLearner(model, tok, LearnerConfig(surprise_gate=False),
                            state_dir=tmp_path / "o")
    learner.observe(["my dog is called Biscuit", "nice"])
    learner.observe(["my dog is called Biscuit", "nice", "I live in Leeds", "lovely"])
    last = list(learner.hippocampus.episodes.values())[-1]
    assert "Leeds" in last and "Biscuit" not in last


def test_hippocampus_can_be_switched_off(model, tok, tmp_path):
    learner = OnlineLearner(model, tok, LearnerConfig(hippocampus_tokens=0),
                            state_dir=tmp_path / "o")
    assert learner.hippocampus is None
    learner.observe(["hello there", "hi"])


def test_generation_with_memory_and_the_recall_report(tmp_path):
    ckpt = create_blank_checkpoint(tmp_path / "b" / "base.pt", size="tiny", block_size=64)
    s = ChatSession(checkpoint=ckpt, max_new_tokens=6)
    s.learner.hippocampus.threshold = -1.0         # recall on any match
    s.turn("my dog is called Biscuit")
    reply = s.reply("what is my dog called?")
    assert isinstance(reply, str)
    assert s.recall_line() and "Biscuit" in s.recall_line()
    assert "hippocampus" in brain_map(s)


# --- cortical areas ---------------------------------------------------------------


def areas_model(k=4, scale=1.0, tok_size=420):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=tok_size, n_layer=2, n_head=4, n_kv_head=2,
                           n_embd=64, block_size=64, n_areas=k, area_scale=scale))


def test_areas_are_no_bigger_than_the_dense_layer():
    dense = areas_model(k=0).num_params()
    areas = areas_model(k=4)
    router = sum(b.ffn.router.weight.numel() for b in areas.blocks)
    assert areas.num_params() - router == dense
    assert areas_model(k=4, scale=2.0).num_params() > dense * 1.3


def test_each_word_goes_to_two_areas_and_usage_is_recorded():
    m = areas_model().eval()
    x = torch.randint(0, 420, (2, 30))
    with torch.no_grad():
        m(x, x)
    usage = m.blocks[0].ffn.usage
    assert float(usage.sum()) == 60                 # every word's first choice
    assert isinstance(m.blocks[0].ffn, Areas)


def test_areas_learn_and_are_kept_balanced():
    m = areas_model().train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 420, (4, 40))
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = m(x, x)[1]
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.7
    assert m.blocks[0].ffn.balance is not None
    for p in m.blocks[0].ffn.router.parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0


def test_a_brain_with_areas_still_grows_exactly():
    m = areas_model().eval()
    x = torch.randint(0, 420, (1, 20))
    with torch.no_grad():
        before = m(x, x)[0]
        grow(m, 1)
        assert torch.equal(m(x, x)[0], before)
    assert isinstance(m.blocks[-1].ffn, Areas)


def test_blank_command_can_make_areas(tmp_path):
    from aria.cli import main
    from aria.pretrain import load_checkpoint
    out = tmp_path / "b.pt"
    assert main(["blank", "--out", str(out), "--size", "tiny", "--areas", "4"]) == 0
    model, *_ = load_checkpoint(out)
    assert model.cfg.n_areas == 4 and isinstance(model.blocks[0].ffn, Areas)
    s = ChatSession(checkpoint=out, max_new_tokens=4)
    assert "specialist areas" in brain_map(s)
    assert isinstance(next(iter(generate(s.model, [1, 3, 5], max_new_tokens=1)), None) or 0, int)
