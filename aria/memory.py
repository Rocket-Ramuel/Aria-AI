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

# Longer than any context window, so nothing that could be trained on is lost;
# short enough that one enormous pasted message can't bloat replay.json.
MAX_STORED_TURN_CHARS = 4_000
# The journal keeps this much recent detail (plus one rotated file of the same
# size); lifetime counts survive rotation in journal_totals.json.
JOURNAL_MAX_BYTES = 4_000_000


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
            "turns": [t[:MAX_STORED_TURN_CHARS] for t in turns],
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
        ), encoding="utf-8")
        tmp.replace(p)   # atomic, so an interrupted save cannot corrupt memory

    @classmethod
    def load(cls, path: str | Path, capacity: int | None = None,
             seed: int = 7) -> "ReplayBuffer":
        p = Path(path)
        buf = cls(capacity=capacity or 4096, seed=seed)
        if not p.exists():
            return buf
        d = json.loads(p.read_text(encoding="utf-8"))
        buf.capacity = capacity or d.get("capacity", 4096)
        buf.seen = d.get("seen", 0)
        buf.items = d.get("items", [])[: buf.capacity]
        return buf


class Journal:
    """Append-only JSONL log of the learner's decisions, bounded on disk.

    Every turn is logged with its text, so a journal left to grow would grow
    forever. Instead, when it passes `max_bytes` it is rotated to
    `journal.1.jsonl` (replacing the previous one), and the counts in the file
    being dropped are folded into `journal_totals.json`. Disk use stays under
    twice `max_bytes`; `summary()` still covers the learner's whole life.
    """

    _COUNTS = ("applied", "skipped", "rollbacks", "consolidations", "loss_n")

    def __init__(self, path: str | Path, max_bytes: int = JOURNAL_MAX_BYTES) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.rotated = self.path.with_name(self.path.stem + ".1" + self.path.suffix)
        self.totals_path = self.path.with_name(self.path.stem + "_totals.json")

    def write(self, **record: Any) -> None:
        record.setdefault("t", time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.path.stat().st_size > self.max_bytes:
            self._rotate()

    def _rotate(self) -> None:
        if self.rotated.exists():
            totals = self._totals()
            dropped = self._count(self._rows(self.rotated))
            for k in self._COUNTS + ("loss_sum",):
                totals[k] = totals.get(k, 0) + dropped[k]
            tmp = self.totals_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(totals), encoding="utf-8")
            tmp.replace(self.totals_path)
        self.path.replace(self.rotated)

    def _totals(self) -> dict[str, float]:
        if self.totals_path.exists():
            return json.loads(self.totals_path.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    @staticmethod
    def _count(rows: list[dict[str, Any]]) -> dict[str, float]:
        updates = [r for r in rows if r.get("event") == "update"]
        losses = [r["loss_after"] for r in updates if r.get("applied") and "loss_after" in r]
        return {
            "applied": sum(1 for r in updates if r.get("applied")),
            "skipped": sum(1 for r in updates if not r.get("applied")),
            "rollbacks": sum(1 for r in rows if r.get("event") == "rollback"),
            "consolidations": sum(1 for r in rows if r.get("event") == "consolidate"),
            "loss_n": len(losses),
            "loss_sum": sum(losses),
        }

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Recent records: the current file, after the rotated one."""
        rows = self._rows(self.rotated) + self._rows(self.path)
        return rows[-limit:] if limit else rows

    def summary(self) -> dict[str, Any]:
        c = self._count(self.read())
        for k, v in self._totals().items():
            c[k] = c.get(k, 0) + v
        return {
            "turns_seen": int(c["applied"] + c["skipped"]),
            "updates_applied": int(c["applied"]),
            "updates_skipped": int(c["skipped"]),
            "rollbacks": int(c["rollbacks"]),
            "consolidations": int(c["consolidations"]),
            "mean_loss_after_update": c["loss_sum"] / c["loss_n"] if c["loss_n"] else None,
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
