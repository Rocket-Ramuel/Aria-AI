"""Choosing the device: NVIDIA GPU, Apple GPU, or CPU, with a safe fallback.

No GPU is needed to run these: GPU presence and failure are simulated, and
the self-test itself is exercised on the CPU."""

import pytest
import torch

from aria import device as dev


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(dev, "_verified", {})


def test_the_self_test_exercises_everything_and_passes_on_cpu():
    assert dev.self_test("cpu") is None


def test_cpu_is_always_cpu():
    assert dev.resolve("cpu") == "cpu"


def test_auto_without_a_gpu_is_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dev, "mps_available", lambda: False)
    assert dev.resolve("auto") == "cpu"


def test_auto_prefers_nvidia_then_apple(monkeypatch):
    monkeypatch.setattr(dev, "self_test", lambda d: None)
    monkeypatch.setattr(dev, "mps_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert dev.resolve("auto") == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dev.resolve("auto") == "mps"


def test_a_small_model_chats_on_a_macs_cpu(monkeypatch):
    """The Apple GPU is slower than the CPU for a small model; only a large one
    (or asking for it) uses it."""
    monkeypatch.setattr(dev, "self_test", lambda d: None)
    monkeypatch.setattr(dev, "mps_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dev.resolve("auto", n_params=6_500_000) == "cpu"
    assert dev.resolve("auto", n_params=dev.MPS_MIN_PARAMS) == "mps"
    assert dev.resolve("mps", n_params=6_500_000) == "mps"
    from aria.chat import ChatSession
    assert ChatSession(max_new_tokens=4).device == "cpu"        # the shipped model


def test_a_gpu_that_fails_its_self_test_falls_back_to_cpu(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dev, "mps_available", lambda: True)
    monkeypatch.setattr(dev, "self_test",
                        lambda d: "NotImplementedError: aten::something on MPS")
    assert dev.resolve("auto") == "cpu"
    assert "Apple GPU failed a self-test" in capsys.readouterr().err
    assert dev.resolve("mps") == "cpu"


def test_asking_for_a_missing_gpu_explains_and_uses_cpu(monkeypatch, capsys):
    monkeypatch.setattr(dev, "mps_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert dev.resolve("mps") == "cpu"
    assert "Apple Silicon" in capsys.readouterr().err
    assert dev.resolve("cuda") == "cpu"


def test_an_unknown_device_is_an_error():
    with pytest.raises(ValueError, match="choose from"):
        dev.resolve("tpu")


def test_a_failing_operation_is_caught_by_the_self_test(monkeypatch):
    def broken(*a, **k):
        raise NotImplementedError("not on this GPU")
    monkeypatch.setattr(torch.Tensor, "scatter_add_", broken)
    assert "not on this GPU" in dev.self_test("cpu")


def test_the_mac_fallback_switch_is_set():
    import os
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_saved_weights_are_on_the_cpu_so_they_load_anywhere(tmp_path):
    from aria.chat import ChatSession
    from aria.pretrain import create_blank_checkpoint
    ckpt = create_blank_checkpoint(tmp_path / "b" / "base.pt", size="tiny", block_size=64)
    # Wherever she runs (an Apple or NVIDIA GPU where there is one), what she
    # saves is on the CPU.
    s = ChatSession(checkpoint=ckpt, max_new_tokens=4)
    s.turn("hello there")
    s.save()
    state = torch.load(s.state_dir / "learned.pt", weights_only=True)
    assert all(v.device.type == "cpu" for v in state["model"].values())
