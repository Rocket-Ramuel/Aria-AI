"""Persistent conversational memory.

Two things live here:

`ReplayBuffer` — a reservoir of past exchanges, stored as plain text so it stays
human-readable and auditable. Rehearsing from it during online learning is what
stops the model from forgetting last week's conversation while learning today's.

`Journal` — an append-only record of every online update: what was learned, how
surprising it was, whether it was applied, and whether it was ever rolled back.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


class ReplayBuffer:
    """Capacity-bounded reservoir of conversations.

    Reservoir sampling keeps the retained set an unbiased sample of the whole
    history, so an old memory is never guaranteed to be evicted just for being
    old. Items can carry a `weight` — corrections are worth rehearsing more than
    small talk — which biases sampling but not retention.
    """

    def __init__(self, capacity: int = 4096, seed: int = 7) -> None:
        self.capacity = capacity
        self.items: list[dict[str, Any]] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, turns: Sequence[str], weight: float = 1.0,
            kind: str = "chat", meta: dict | None = None) -> bool:
        """Insert an exchange. Returns True if it was retained."""
        if len(turns) < 2:
            return False
        item = {
            "turns": list(turns),
            "weight": float(weight),
            "kind": kind,
            "t": time.time(),
            "meta": meta or {},
        }
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return True
        j = self.rng.randrange(self.seen)
        if j < self.capacity:
            self.items[j] = item
            return True
        return False

    def sample(self, n: int) -> list[dict[str, Any]]:
        if not self.items or n <= 0:
            return []
        weights = [max(1e-6, it["weight"]) for it in self.items]
        return self.rng.choices(self.items, weights=weights, k=min(n, len(self.items) * 4))

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        return sorted(self.items, key=lambda it: it["t"], reverse=True)[:n]

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(
            {"capacity": self.capacity, "seen": self.seen, "items": self.items},
            ensure_ascii=False,
        ))
        tmp.replace(p)   # atomic, so an interrupted save cannot corrupt memory

    @classmethod
    def load(cls, path: str | Path, capacity: int | None = None,
             seed: int = 7) -> "ReplayBuffer":
        p = Path(path)
        buf = cls(capacity=capacity or 4096, seed=seed)
        if not p.exists():
            return buf
        d = json.loads(p.read_text())
        buf.capacity = capacity or d.get("capacity", 4096)
        buf.seen = d.get("seen", 0)
        buf.items = d.get("items", [])[: buf.capacity]
        return buf


class Journal:
    """Append-only JSONL log of the learner's decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **record: Any) -> None:
        record.setdefault("t", time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line]
        return rows[-limit:] if limit else rows

    def summary(self) -> dict[str, Any]:
        rows = self.read()
        applied = [r for r in rows if r.get("event") == "update" and r.get("applied")]
        skipped = [r for r in rows if r.get("event") == "update" and not r.get("applied")]
        rollbacks = [r for r in rows if r.get("event") == "rollback"]
        consolidations = [r for r in rows if r.get("event") == "consolidate"]
        losses = [r["loss_after"] for r in applied if "loss_after" in r]
        return {
            "turns_seen": len(applied) + len(skipped),
            "updates_applied": len(applied),
            "updates_skipped": len(skipped),
            "rollbacks": len(rollbacks),
            "consolidations": len(consolidations),
            "mean_loss_after_update": sum(losses) / len(losses) if losses else None,
        }


def canary_texts(extra: Iterable[str] = ()) -> list[str]:
    """Fixed English probes used to detect degradation.

    These are never trained on. If the model's loss on them starts climbing
    while it learns from a conversation, it is forgetting English, and the
    learner rolls back."""
    base = [
        "The quick brown fox jumps over the lazy dog.",
        "She opened the door and stepped out into the cold morning air.",
        "In the beginning the universe was very hot and very dense.",
        "He said that he would arrive before the end of the week.",
        "Water freezes at zero degrees Celsius and boils at one hundred.",
        "The book on the table belongs to my sister, not to me.",
        "They walked along the river until the light began to fade.",
        "It is difficult to explain, but I will try to make it clear.",
        "The company announced that profits had risen for the third year.",
        "If you heat the metal, it expands; if you cool it, it contracts.",
        "There are seven days in a week and twelve months in a year.",
        "A large crowd gathered in the square to listen to the speech.",
    ]
    return base + list(extra)
