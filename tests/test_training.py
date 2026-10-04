"""Unit tests for pretraining details: schedule, Fisher, batching."""

import random

import numpy as np
import torch

from aria.config import ModelConfig, TrainConfig
from aria.data import ChatSet, TokenStream, mixed_batch
from aria.model import GPT
from aria.pretrain import estimate_fisher, lr_at


def test_cosine_reaches_its_floor_at_the_horizon():
    cfg = TrainConfig(learning_rate=1e-3, min_lr_frac=0.1, warmup_steps=10,
                      max_steps=10_000)
    floor = 1e-4
    # Without a horizon, step 1000 of 10000 is still near the peak ...
    assert lr_at(1000, cfg) > 0.9e-3
    # ... but a time budget that ends at step 1000 should be at the floor.
    assert abs(lr_at(1000, cfg, horizon=1000) - floor) < 1e-9
    assert lr_at(500, cfg, horizon=1000) < lr_at(500, cfg)
    # A horizon past max_steps is clamped to it.
    assert lr_at(10_000, cfg, horizon=50_000) == lr_at(10_000, cfg)


def test_fisher_is_the_mean_of_per_sequence_squared_gradients(tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, n_layer=1, n_head=2, n_kv_head=1,
                      n_embd=16, block_size=8)
    model = GPT(cfg)
    np.random.default_rng(0).integers(0, 50, 400).astype(np.uint16).tofile(tmp_path / "t.bin")
    stream = TokenStream(tmp_path / "t.bin", 8)

    fisher = estimate_fisher(model, stream, batches=2, batch_size=3,
                             generator=torch.Generator().manual_seed(1))

    # Recompute by hand from the same crops.
    g = torch.Generator().manual_seed(1)
    crops = [stream.batch(3, g) for _ in range(2)]
    name = "blocks.0.ffn.up_proj.weight"
    p = dict(model.named_parameters())[name]
    expected = torch.zeros_like(p)
    for x, y in crops:
        for j in range(3):
            model.zero_grad()
            model(x[j:j + 1], y[j:j + 1])[1].backward()
            expected += p.grad ** 2
    expected /= 6
    assert torch.allclose(fisher[name], expected, rtol=1e-4, atol=1e-10)

    # And it is not the square of the batch-mean gradient, which is smaller.
    model.zero_grad()
    x, y = crops[0]
    model(x, y)[1].backward()
    assert fisher[name].sum() > (p.grad ** 2).sum()


def test_mixed_batch_crops_chat_examples_to_the_model(tmp_path):
    import pickle
    long_x = list(range(1, 201))
    long_y = [-100] * 150 + list(range(2, 52))
    with open(tmp_path / "chat.pt", "wb") as f:
        pickle.dump([(long_x, long_y)], f)
    chat = ChatSet(tmp_path / "chat.pt", pad_id=0)
    x, y = mixed_batch(None, chat, 4, 1.0, random.Random(0), None, 0, block_size=64)
    assert x.shape[1] == 64
    # The tail survives: that is where the reply and its loss are.
    assert x[0, -1].item() == 200 and (y[0] != -100).sum() == 50
