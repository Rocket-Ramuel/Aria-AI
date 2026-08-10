import math

import pytest
import torch

from aria.config import ModelConfig
from aria.model import (
    GPT, IGNORE_INDEX, LoRALinear, attach_lora, lora_parameters, merge_lora,
)
from aria.sample import generate

CFG = ModelConfig(vocab_size=128, n_layer=2, n_head=4, n_kv_head=2,
                  n_embd=32, block_size=32)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return GPT(CFG)


def test_forward_shapes_and_loss():
    m = tiny_model()
    x = torch.randint(0, CFG.vocab_size, (3, 16))
    logits, loss, _ = m(x, x)
    assert logits.shape == (3, 16, CFG.vocab_size)
    # An untrained model should sit near uniform entropy.
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.8


def test_attention_is_causal():
    """Changing a later token must not change an earlier position's logits."""
    m = tiny_model().eval()
    x = torch.randint(0, CFG.vocab_size, (1, 12))
    a, _, _ = m(x, x)
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % CFG.vocab_size
    b, _, _ = m(x2, x2)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)
    assert not torch.allclose(a[0, -1], b[0, -1], atol=1e-5)


def test_kv_cache_matches_full_forward():
    m = tiny_model().eval()
    x = torch.randint(0, CFG.vocab_size, (1, 10))
    full, _, _ = m(x)                      # logits for the last position only

    caches = m.empty_cache()
    out, _, caches = m(x[:, :1], kv_caches=caches)
    for i in range(1, 10):
        out, _, caches = m(x[:, i : i + 1], kv_caches=caches)
    assert torch.allclose(full[:, -1], out[:, -1], atol=1e-4)


def test_ignore_index_masks_loss():
    m = tiny_model()
    x = torch.randint(0, CFG.vocab_size, (2, 8))
    y = x.clone()
    y[:, :4] = IGNORE_INDEX
    _, masked, _ = m(x, y)
    _, full, _ = m(x, x)
    assert not torch.isnan(masked)
    assert masked.item() != pytest.approx(full.item())


def test_block_size_is_enforced():
    m = tiny_model()
    x = torch.randint(0, CFG.vocab_size, (1, CFG.block_size + 1))
    with pytest.raises(ValueError):
        m(x, x)


def test_generate_respects_stop_and_length():
    m = tiny_model().eval()
    out = list(generate(m, [1, 2, 3], max_new_tokens=7, temperature=1.0,
                        top_k=None, top_p=None, repetition_penalty=1.0))
    assert len(out) <= 7
    assert all(0 <= t < CFG.vocab_size for t in out)


def test_generate_handles_context_overflow():
    m = tiny_model().eval()
    out = list(generate(m, list(range(CFG.block_size - 2)), max_new_tokens=40,
                        temperature=1.0, top_k=None, top_p=None))
    assert len(out) == 40


def test_grouped_query_attention_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=30, n_head=4)
    with pytest.raises(ValueError):
        ModelConfig(n_head=4, n_kv_head=3)


def test_tied_embeddings_share_storage():
    m = tiny_model()
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


# --- LoRA -----------------------------------------------------------------


def test_fresh_lora_is_a_noop():
    m = tiny_model().eval()
    x = torch.randint(0, CFG.vocab_size, (1, 8))
    before, _, _ = m(x)
    n = attach_lora(m, rank=4, alpha=8.0)
    assert n > 0
    after, _, _ = m.eval()(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_only_lora_params_are_trainable():
    m = tiny_model()
    attach_lora(m, rank=4)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in lora_parameters(m):
        p.requires_grad_(True)
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)


def test_merge_lora_preserves_function():
    m = tiny_model().eval()
    attach_lora(m, rank=4, alpha=8.0)
    with torch.no_grad():
        for p in lora_parameters(m):
            p.add_(torch.randn_like(p) * 0.05)

    x = torch.randint(0, CFG.vocab_size, (1, 8))
    before, _, _ = m(x)
    n = merge_lora(m)
    assert n > 0
    after, _, _ = m.eval()(x)
    assert torch.allclose(before, after, atol=1e-4)
    assert not any(isinstance(mod, LoRALinear) for mod in m.modules())


def test_merged_model_loads_as_plain_checkpoint():
    m = tiny_model()
    attach_lora(m, rank=4)
    with torch.no_grad():
        for p in lora_parameters(m):
            p.add_(torch.randn_like(p) * 0.05)
    merge_lora(m)
    plain = GPT(CFG)
    plain.load_state_dict(m.state_dict())   # must be a vanilla state dict


def test_model_can_overfit_a_sequence():
    """The clearest end-to-end check that gradients flow correctly."""
    m = tiny_model(seed=3)
    x = torch.randint(0, CFG.vocab_size, (1, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(120):
        opt.zero_grad()
        _, loss, _ = m(x, x)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.2
