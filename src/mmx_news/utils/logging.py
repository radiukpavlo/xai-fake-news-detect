from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .io import dump_json


def runtime_fingerprint() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["python"] = sys.version
    info["platform"] = platform.platform()
    info["processor"] = platform.processor()
    info["machine"] = platform.machine()
    info["python_executable"] = sys.executable
    # package versions (best-effort)
    try:
        import importlib.metadata as md  # Py3.8+
    except Exception:
        md = None
    if md is not None:
        pkgs = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "sentence-transformers", "torch", "tqdm", "umap-learn", "pyyaml"]
        versions = {}
        for p in pkgs:
            try:
                versions[p] = md.version(p)
            except Exception:
                pass
        info["packages"] = versions
    # pip freeze (best-effort)
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=10)
        info["pip_freeze"] = out.splitlines()
    except Exception:
        info["pip_freeze"] = []
    return info


def save_run_provenance(path: str | Path, cfg: Dict[str, Any], run_id: str) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "config": cfg,
        "fingerprint": runtime_fingerprint(),
    }
    dump_json(record, p / "run.json")
