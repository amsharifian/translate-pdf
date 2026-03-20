from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def discover_pdfs(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if "*" in item or "?" in item or "[" in item:
            paths.extend(Path().glob(item))
        elif p.is_dir():
            paths.extend(p.rglob("*.pdf"))
        else:
            paths.append(p)
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


__all__ = ["discover_pdfs"]
