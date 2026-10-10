"""Aria — a small English language model that keeps learning as you talk to it.

The model, tokenizer, training loop and continual-learning machinery are all
implemented here from scratch on top of PyTorch tensors; nothing pretrained is
downloaded.

The names below are imported on first use, so `import aria` doesn't load
PyTorch: the app shows its window first and loads the model behind it.
"""

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    "AriaConfig": "config", "LearnerConfig": "config", "ModelConfig": "config",
    "TrainConfig": "config", "preset": "config",
    "GPT": "model", "attach_lora": "model", "merge_lora": "model",
    "BPETokenizer": "tokenizer",
    "OnlineLearner": "learner", "UpdateReport": "learner",
    "resume_learned_weights": "learner",
    "ReplayBuffer": "memory", "Journal": "memory",
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name: str):
    if name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
