from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from ..utils.io import config_hash, load_yaml
from ..utils.logging import save_run_provenance
from ..utils.seeds import set_all_seeds


def load_config(path: str | Path) -> Dict[str, Any]:
    return load_yaml(path)


def resolve_run(cfg: Dict[str, Any]) -> Tuple[str, Path]:
    run_id = config_hash(cfg)[:10]
    out_dir = Path("runs") / cfg.get("run_name", "exp") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_id, out_dir


def init_run(cfg: Dict[str, Any]) -> Tuple[str, Path]:
    seed = int(cfg.get("repro", {}).get("global_seed", 42))
    deterministic = bool(cfg.get("repro", {}).get("deterministic", True))
    set_all_seeds(seed, deterministic=deterministic)
    run_id, out_dir = resolve_run(cfg)
    save_run_provenance(out_dir, cfg, run_id)
    return run_id, out_dir


def train_val_test_arrays(X: np.ndarray, idx: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return X[idx["train_idx"]], X[idx["val_idx"]], X[idx["test_idx"]]
