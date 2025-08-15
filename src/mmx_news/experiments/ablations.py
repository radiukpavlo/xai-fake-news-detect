from __future__ import annotations

from pathlib import Path

from ..training.train import run_training
from ..training.utils import load_config
from ..utils.io import load_yaml


def run_ablations(config_path: str | Path, grid_path: str | Path) -> None:
    cfg = load_config(config_path)
    grid = load_yaml(grid_path)
    kinds = grid.get("classifier_type", ["linear_svm"])
    weights_list = grid.get("embeddings_weights", [cfg["embeddings"]["weights"]])

    for kind in kinds:
        for weights in weights_list:
            cfg["classifier"]["type"] = kind
            cfg["embeddings"]["weights"] = weights
            # Use the first experiment seed only for brevity (extend as needed)
            seed = int(cfg["data"]["seed_list"][0])
            run_training(cfg, seed)
