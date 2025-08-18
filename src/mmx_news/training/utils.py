from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..utils.config import Config
from ..utils.io import config_hash, load_yaml
from ..utils.logging import save_run_provenance
from ..utils.seeds import set_all_seeds


def load_config(path: str | Path) -> Config:
    cfg_dict = load_yaml(path)
    return Config(**cfg_dict)


def resolve_run(cfg: Config, out_dir: Path | None = None) -> Tuple[str, Path]:
    run_id = config_hash(cfg.model_dump())[:10]
    if out_dir is None:
        out_dir = Path("runs") / cfg.run_name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_id, out_dir


def init_run(cfg: Config, out_dir: Path | None = None) -> Tuple[str, Path]:
    seed = cfg.repro.global_seed
    deterministic = cfg.repro.deterministic
    set_all_seeds(seed, deterministic=deterministic)
    run_id, out_dir = resolve_run(cfg, out_dir=out_dir)
    save_run_provenance(out_dir, cfg.model_dump(), run_id)
    return run_id, out_dir


def train_val_test_arrays(X: np.ndarray, idx: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return X[idx["train_idx"]], X[idx["val_idx"]], X[idx["test_idx"]]
