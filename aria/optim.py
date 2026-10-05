"""A low-memory Adam for learning in a large model.

AdamW keeps two float32 numbers per weight — 8 bytes, twice the weights
themselves. For a 100M-parameter model learning with full plasticity that is
800 MB of optimiser state alone. This optimiser keeps about 2 bytes per weight:

* the **first moment** (momentum) is stored in bfloat16 — it is a running
  average of gradients, and its low bits are noise anyway;
* the **second moment** of every matrix is **factored** (as in Adafactor):
  instead of a value per entry, one running mean per row and one per column,
  whose outer product estimates the full matrix. For a 768 x 2048 weight that
  is 2,816 numbers instead of 1.5 million. Vectors (norm gains) are tiny and
  keep an exact second moment.

The update is otherwise Adam's, with bias correction, so learning rates
carry over. It deliberately does not subclass `torch.optim.Optimizer`: the
first instance of that base class makes PyTorch import its compiler stack,
about 150 MB of resident memory the learner has no use for.
"""

from __future__ import annotations

from typing import Iterable

import torch


class LowMemoryAdam:
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.99), eps: float = 1e-8) -> None:
        self.param_groups = [{"params": list(params), "lr": lr}]
        self.betas = betas
        self.eps = eps
        self.state: dict[torch.nn.Parameter, dict] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.param_groups[0]["params"]:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for st in self.state.values() for t in st.values()
                   if isinstance(t, torch.Tensor))

    @torch.no_grad()
    def step(self) -> None:
        b1, b2 = self.betas
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                g = g.float()
                st = self.state.get(p)
                if st is None:
                    # bfloat16 on CPUs and NVIDIA GPUs; float32 on an Apple GPU,
                    # where bfloat16 needs a recent macOS.
                    m_dtype = torch.float32 if p.device.type == "mps" else torch.bfloat16
                    st = self.state[p] = {"t": 0, "m": torch.zeros_like(p, dtype=m_dtype)}
                    if p.dim() >= 2:
                        st["row"] = torch.zeros(p.shape[:-1], dtype=torch.float32, device=p.device)
                        st["col"] = torch.zeros(p.shape[:-2] + p.shape[-1:],
                                                dtype=torch.float32, device=p.device)
                    else:
                        st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["t"] += 1
                t = st["t"]

                m = st["m"].float().mul_(b1).add_(g, alpha=1 - b1)
                st["m"].copy_(m)

                g2 = g.square().add_(1e-30)
                if p.dim() >= 2:
                    st["row"].mul_(b2).add_(g2.mean(-1), alpha=1 - b2)
                    st["col"].mul_(b2).add_(g2.mean(-2), alpha=1 - b2)
                    row, col = st["row"], st["col"]
                    v = (row.unsqueeze(-1) * col.unsqueeze(-2)
                         / row.mean(-1, keepdim=True).unsqueeze(-1).clamp_min(1e-30))
                else:
                    st["v"].mul_(b2).add_(g2, alpha=1 - b2)
                    v = st["v"]

                m_hat = m.div_(1 - b1 ** t)
                denom = v.div(1 - b2 ** t).sqrt_().add_(self.eps)
                p.add_((m_hat / denom).to(p.dtype), alpha=-lr)
