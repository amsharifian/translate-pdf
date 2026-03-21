"""Translation Memory — sentence/paragraph-level feedback store with TF-IDF retrieval.

Stores corrected translation pairs and injects the most relevant ones as few-shot
examples when translating new text.  The feedback loop:
translate → review → correct → store → improve.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


DEFAULT_TM_PATH = Path("jobs/translation_memory.jsonl")

# Retrieval defaults
DEFAULT_TOP_K = 3
MIN_SIMILARITY = 0.1
MAX_EXAMPLE_CHARS = 3200  # ~800 tokens at 4 chars/token


@dataclass
class TMEntry:
    source: str
    target: str
    target_lang: str = "fa"
    context: str = ""
    created_at: str = ""
    source_job: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ── TF-IDF helpers (pure Python, no external deps) ─────────────

_SPLIT_RE = re.compile(r"[\w\u0600-\u06FF\u0750-\u077F]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenisation supporting Latin + Arabic/Persian scripts."""
    return [m.group().lower() for m in _SPLIT_RE.finditer(text)]


def _tf(tokens: List[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def _idf(corpus_tfs: List[dict[str, float]]) -> dict[str, float]:
    n = len(corpus_tfs) or 1
    df: Counter[str] = Counter()
    for tf_map in corpus_tfs:
        df.update(tf_map.keys())
    return {term: math.log((n + 1) / (freq + 1)) + 1 for term, freq in df.items()}


def _tfidf_vector(tf_map: dict[str, float], idf_map: dict[str, float]) -> dict[str, float]:
    return {t: tf_val * idf_map.get(t, 0) for t, tf_val in tf_map.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Main class ──────────────────────────────────────────────────


class TranslationMemory:
    """Append-only JSONL store with TF-IDF retrieval."""

    def __init__(self, path: Path | str = DEFAULT_TM_PATH) -> None:
        self.path = Path(path)
        self._entries: List[TMEntry] | None = None  # lazy-loaded cache

    # ── CRUD ────────────────────────────────────────────────────

    def add(self, entry: TMEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        # Invalidate cache
        self._entries = None

    def load_all(self, target_lang: str | None = None) -> List[TMEntry]:
        if self._entries is None:
            self._entries = self._read_file()
        entries = self._entries
        if target_lang:
            entries = [e for e in entries if e.target_lang == target_lang]
        return entries

    def count(self, target_lang: str | None = None) -> int:
        return len(self.load_all(target_lang))

    def delete(self, index: int) -> bool:
        """Delete entry at *index* (0-based). Rewrites the file."""
        entries = self.load_all()
        if index < 0 or index >= len(entries):
            return False
        entries.pop(index)
        self._write_all(entries)
        return True

    def update(self, index: int, entry: TMEntry) -> bool:
        entries = self.load_all()
        if index < 0 or index >= len(entries):
            return False
        entries[index] = entry
        self._write_all(entries)
        return True

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self._entries = None

    # ── Search ──────────────────────────────────────────────────

    def search(
        self,
        query: str,
        target_lang: str = "fa",
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = MIN_SIMILARITY,
        max_chars: int = MAX_EXAMPLE_CHARS,
    ) -> List[tuple[TMEntry, float]]:
        """Return the *top_k* most similar TM entries above *min_similarity*.

        Results are capped at *max_chars* total (source+target combined) to
        stay within token budgets.
        """
        entries = self.load_all(target_lang)
        if not entries:
            return []

        # Build corpus TF vectors
        corpus_tokens = [_tokenize(e.source) for e in entries]
        corpus_tfs = [_tf(t) for t in corpus_tokens]
        idf_map = _idf(corpus_tfs)

        # Query vector
        q_tf = _tf(_tokenize(query))
        q_vec = _tfidf_vector(q_tf, idf_map)

        scored: List[tuple[int, float]] = []
        for i, tf_map in enumerate(corpus_tfs):
            vec = _tfidf_vector(tf_map, idf_map)
            sim = _cosine(q_vec, vec)
            if sim >= min_similarity:
                scored.append((i, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[tuple[TMEntry, float]] = []
        chars_used = 0
        for idx, sim in scored[:top_k]:
            entry = entries[idx]
            entry_chars = len(entry.source) + len(entry.target)
            if chars_used + entry_chars > max_chars:
                break
            results.append((entry, sim))
            chars_used += entry_chars

        return results

    # ── Export / Import ─────────────────────────────────────────

    def export_json(self) -> str:
        entries = self.load_all()
        return json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=2)

    def import_json(self, data: str | list) -> int:
        """Import entries from JSON string or list of dicts. Returns count imported."""
        if isinstance(data, str):
            data = json.loads(data)
        count = 0
        for item in data:
            try:
                entry = TMEntry(**item)
                self.add(entry)
                count += 1
            except (TypeError, KeyError):
                continue
        return count

    # ── Internals ───────────────────────────────────────────────

    def _read_file(self) -> List[TMEntry]:
        if not self.path.exists():
            return []
        entries: List[TMEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append(TMEntry(**d))
            except (json.JSONDecodeError, TypeError):
                continue
        return entries

    def _write_all(self, entries: List[TMEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
        self._entries = entries
