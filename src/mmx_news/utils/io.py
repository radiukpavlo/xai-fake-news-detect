from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True, ensure_ascii=False)


def config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
