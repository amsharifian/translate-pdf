from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def _get_nested(cfg: Dict[str, Any], *keys: str) -> Optional[Any]:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def get_openai_config(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    return {
        "api_key": _get_nested(cfg, "openai", "api_key"),
        "model": _get_nested(cfg, "openai", "model"),
        "base_url": _get_nested(cfg, "openai", "base_url"),
    }


def get_default_font_path(cfg: Dict[str, Any]) -> Optional[str]:
    return _get_nested(cfg, "font", "default_path")
