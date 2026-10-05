"""Aria's hippocampus: memory from a single experience.

A brain learns in two ways at once (the "complementary learning systems"
view of memory). The neocortex learns slowly, by small changes over many
repetitions, and builds general knowledge. The hippocampus stores a specific
experience after one exposure, recalls it when something similar comes up,
and replays it to the cortex during sleep until the cortex knows it too.

Aria's weights are her cortex: one gradient step per exchange barely moves
the odds of a fact she has been told once. This module is the fast system.
Every exchange she hears is stored as a run of *keys* — the cortex's internal
state at each word — each paired with the word that came next. When she
speaks, her current state is compared with the stored keys; if it closely
matches a remembered moment, what came next in that moment is blended into
her prediction (a nearest-neighbour language model). Tell her your dog is
called Biscuit once, ask later, and the memory supplies "Biscuit".

* **Small:** keys are float16 and the store is capped (16,384 words by
  default: 8 MB for the small model). The memories themselves are the replay
  buffer's text, which is already saved, so nothing extra goes to disk.
* **Reconsolidation:** as the cortex learns, its internal states change, so
  stored keys go stale. During sleep (consolidation) every memory is
  re-encoded with the current cortex — memory is rebuilt, not just read.
* **Episode first, then detail.** A small model's internal state at a word
  says mostly "a name comes next", not *which* conversation this is, so with
  thousands of memories the wrong "...is called" wins. The hippocampus
  handles this by recalling the right *episode* from contextual cues before
  the details within it. Here the cues are the conversation's words, weighted
  by how rare they are across memories (TF-IDF): "dog" and "Wexford" count,
  "my" and "is" hardly at all. Word-level recall then searches only the few
  episodes that match.
* **Gated:** recall only takes part when an episode is on topic and a moment
  in it matches closely; ordinary text is predicted by the cortex alone.
"""

from __future__ import annotations

import math

from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from .data import collate, encode_dialogue
from .model import IGNORE_INDEX


class Hippocampus:
    def __init__(self, model, tok, capacity_tokens: int = 16_384, k: int = 8,
                 threshold: float = 0.35, strength: float = 0.5,
                 temperature: float = 0.05, device: str = "cpu",
                 episodes_considered: int = 8, min_episode_score: float = 0.2) -> None:
        self.model, self.tok = model, tok
        self.capacity = capacity_tokens
        self.k, self.threshold, self.strength = k, threshold, strength
        self.temperature = temperature
        self.device = device
        d = model.cfg.n_embd
        self.keys = torch.empty(0, d, dtype=torch.float16, device=device)
        self.values = torch.empty(0, dtype=torch.long, device=device)
        self.owner = torch.empty(0, dtype=torch.long, device=device)   # episode id per key
        self.episodes: dict[int, str] = {}       # id -> short description
        self.cues: dict[int, frozenset[int]] = {}   # id -> its distinct tokens
        self.df: dict[int, int] = {}             # token -> episodes containing it
        self.episodes_considered = episodes_considered
        self.min_episode_score = min_episode_score
        self.focus_mask: torch.Tensor | None = None   # keys of the recalled episodes
        self._next_id = 0
        self.recalls = 0                         # predictions a memory took part in
        self.last_recall: tuple[float, str] | None = None

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def memory_mb(self) -> float:
        return (self.keys.numel() * 2 + self.values.numel() * 16) / 1e6

    # -- storing ---------------------------------------------------------

    def _sequence(self, turns: Sequence[str], kind: str) -> list[int] | None:
        """The token sequence an experience is remembered as."""
        if kind == "document":
            ids = [self.tok.bos_id, self.tok.aria_id] + self.tok.encode(
                " " + turns[-1].strip(), allowed_special=False) + [self.tok.eot_id]
            return ids[: self.model.cfg.block_size + 1]
        enc = encode_dialogue(self.tok, turns, self.model.cfg.block_size)
        if enc is None:
            return None
        x, y = enc
        return list(x) + [y[-1] if y[-1] != IGNORE_INDEX else self.tok.eot_id]

    @torch.no_grad()
    def _encode(self, seqs: list[list[int]]) -> list[torch.Tensor]:
        """Unit-length keys for every position of each sequence."""
        was_training = self.model.training
        self.model.eval()
        out = []
        for i in range(0, len(seqs), 16):
            chunk = seqs[i : i + 16]
            x, _ = collate([(s[:-1], s[1:]) for s in chunk], self.tok.pad_id)
            x = x.to(self.device)
            hidden = self.model(x, x, return_hidden=True)[3].float()
            for s, h in zip(chunk, hidden):
                out.append(F.normalize(h[: len(s) - 1], dim=-1).half())
        if was_training:
            self.model.train()
        return out

    def _add(self, seqs: list[list[int]], labels: list[str]) -> None:
        keys = self._encode(seqs)
        new_k, new_v, new_o = [], [], []
        for s, k, label in zip(seqs, keys, labels):
            eid = self._next_id
            self._next_id += 1
            self.episodes[eid] = label
            cues = frozenset(t for t in s if t >= self.tok.n_special)
            self.cues[eid] = cues
            for t in cues:
                self.df[t] = self.df.get(t, 0) + 1
            new_k.append(k)
            new_v.append(torch.tensor(s[1:], dtype=torch.long, device=self.device))
            new_o.append(torch.full((len(s) - 1,), eid, dtype=torch.long, device=self.device))
        if not new_k:
            return
        self.keys = torch.cat([self.keys] + new_k)
        self.values = torch.cat([self.values] + new_v)
        self.owner = torch.cat([self.owner] + new_o)
        self._forget_oldest()

    def _forget_oldest(self) -> None:
        excess = len(self) - self.capacity
        if excess <= 0:
            return
        self.keys, self.values = self.keys[excess:], self.values[excess:]
        self.owner = self.owner[excess:]
        alive = set(self.owner.unique().tolist())
        for eid in [e for e in self.episodes if e not in alive]:
            del self.episodes[eid]
            for t in self.cues.pop(eid, ()):
                self.df[t] -= 1
                if not self.df[t]:
                    del self.df[t]
        self.focus_mask = None

    def store(self, turns: Sequence[str], kind: str = "chat") -> None:
        """Remember one experience (an exchange, or a passage of a document)."""
        seq = self._sequence(turns, kind)
        if seq is not None and len(seq) > 2:
            self._add([seq], [_describe(turns, kind)])

    def rebuild(self, items: Iterable[dict[str, Any]]) -> None:
        """Re-encode remembered experiences with the cortex as it is now.

        Called at load and during sleep. `items` are replay-buffer entries;
        the most recent are kept, up to the capacity."""
        newest = sorted(items, key=lambda it: it.get("t", 0), reverse=True)
        seqs, labels, total = [], [], 0
        for it in newest:
            kind = it.get("kind", "chat")
            # Replay keeps an exchange with its context; the episode is the
            # exchange itself (as in `store`).
            turns = it["turns"] if kind == "document" else it["turns"][-2:]
            seq = self._sequence(turns, kind)
            if seq is None or len(seq) <= 2:
                continue
            if total + len(seq) - 1 > self.capacity:
                break
            seqs.append(seq)
            labels.append(_describe(turns, kind))
            total += len(seq) - 1
        d = self.keys.shape[1]
        self.keys = torch.empty(0, d, dtype=torch.float16, device=self.device)
        self.values = torch.empty(0, dtype=torch.long, device=self.device)
        self.owner = torch.empty(0, dtype=torch.long, device=self.device)
        self.episodes, self.cues, self.df = {}, {}, {}
        self.focus_mask = None
        self._add(seqs[::-1], labels[::-1])           # oldest first, newest last

    # -- recalling -------------------------------------------------------

    def focus(self, context_ids: Sequence[int]) -> list[tuple[float, str]]:
        """Recall the episodes the conversation is about; word-level recall
        is then limited to them. Returns [(score, description)], best first.

        Call once per reply with the recent conversation."""
        n = len(self.episodes)
        query = {t for t in context_ids if t >= self.tok.n_special}
        if not n or not query:
            self.focus_mask = torch.zeros(len(self), dtype=torch.bool, device=self.device)
            return []
        # Words in no memory can't tell one episode from another, so they
        # neither help nor dilute the score.
        # Nor do cues found in most memories (stop-words, and the fragments
        # an unfamiliar word breaks into): they can't tell episodes apart.
        common = max(1, n // 2)
        query = {t for t in query if 0 < self.df.get(t, 0) <= common}
        idf = {t: math.log((n + 1) / self.df[t]) for t in query}
        total = sum(idf.values()) or 1.0
        scored = []
        for eid, cues in self.cues.items():
            shared = query & cues
            if not shared:
                continue
            # How much of the cue's information this episode accounts for:
            # long and short episodes are judged alike.
            score = sum(idf[t] for t in shared) / total
            if score >= self.min_episode_score:
                scored.append((score, eid))
        scored.sort(reverse=True)
        chosen = [eid for _, eid in scored[: self.episodes_considered]]
        self.focus_mask = torch.isin(self.owner, torch.tensor(chosen, dtype=torch.long,
                                                              device=self.device))
        return [(sc, self.episodes[eid]) for sc, eid in scored[: self.episodes_considered]]

    @torch.no_grad()
    def recall(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Blend remembered continuations into the cortex's prediction.

        hidden: (T, d) final states; logits: (T, V). Returns log-probabilities
        (T, V). Where no memory is close, it is the cortex's own prediction."""
        logp = F.log_softmax(logits.float(), dim=-1)
        mask = self.focus_mask
        if len(self) == 0 or mask is None or not bool(mask.any()):
            return logp
        keys, values, owner = self.keys[mask], self.values[mask], self.owner[mask]
        q = F.normalize(hidden.float(), dim=-1)
        out = torch.empty_like(logp)
        for i in range(0, q.shape[0], 64):            # bounded memory per call
            qi = q[i : i + 64]
            sims = qi @ keys.float().T
            top, idx = sims.topk(min(self.k, sims.shape[1]), dim=-1)
            w = F.softmax(top / self.temperature, dim=-1)
            p_mem = torch.zeros_like(logp[i : i + 64]).scatter_add_(1, values[idx], w)
            gate = (self.strength * ((top[:, :1] - self.threshold)
                                     / (1 - self.threshold)).clamp(0, 1))
            p = (1 - gate) * logp[i : i + 64].exp() + gate * p_mem
            out[i : i + 64] = p.clamp_min(1e-12).log()
            best = int(top[:, 0].argmax())
            if float(gate[best]) > 0:
                self.recalls += int((gate > 0).sum())
                sim = float(top[best, 0])
                if self.last_recall is None or sim > self.last_recall[0]:
                    eid = int(owner[idx[best, 0]])
                    self.last_recall = (sim, self.episodes.get(eid, ""))
        return out


def _describe(turns: Sequence[str], kind: str) -> str:
    text = turns[-1] if kind == "document" else " / ".join(turns[-2:])
    text = " ".join(text.split())
    return text if len(text) <= 90 else text[:87] + "..."
