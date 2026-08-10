"""Aria — a small English language model that keeps learning as you talk to it.

The model, tokenizer, training loop and continual-learning machinery are all
implemented here from scratch on top of PyTorch tensors; nothing pretrained is
downloaded.
"""

from .config import AriaConfig, LearnerConfig, ModelConfig, TrainConfig, preset
from .learner import OnlineLearner, UpdateReport, resume_learned_weights
from .memory import Journal, ReplayBuffer
from .model import GPT, attach_lora, merge_lora
from .tokenizer import BPETokenizer

__version__ = "0.1.0"

__all__ = [
    "AriaConfig", "LearnerConfig", "ModelConfig", "TrainConfig", "preset",
    "GPT", "BPETokenizer", "OnlineLearner", "UpdateReport",
    "ReplayBuffer", "Journal", "attach_lora", "merge_lora",
    "resume_learned_weights", "__version__",
]
