from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
from typing import Optional


def resolve_font_path(font_arg: str) -> Optional[str]:
    # Direct path
    p = Path(font_arg).expanduser()
    if p.exists() and p.is_file():
        return str(p)

    # Try fontconfig if available
    if shutil.which("fc-list"):
        try:
            output = subprocess.check_output(
                ["fc-list", "-f", "%{file}::%{family}\n"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8", errors="ignore")
            needle = font_arg.lower()
            for line in output.splitlines():
                if "::" not in line:
                    continue
                file_path, family = line.split("::", 1)
                if needle in file_path.lower() or needle in family.lower():
                    return file_path
        except Exception:
            pass

    # Fallback to common font directories (macOS)
    candidates = [
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
    ]

    needle = font_arg.lower()
    for base in candidates:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for filename in files:
                lower = filename.lower()
                if not lower.endswith((".ttf", ".otf", ".ttc")):
                    continue
                if needle in lower:
                    return str(Path(root) / filename)

    return None
