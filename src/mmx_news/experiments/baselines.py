from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..training.train import run_training
from ..training.utils import load_config


def run_baselines(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    seeds = list(cfg["data"]["seed_list"])[: int(cfg["data"]["splits"])]
    kinds = ["linear_svm", "logreg", "rbf_svm"]
    rows = []
    for kind in kinds:
        cfg["classifier"]["type"] = kind
        for seed in seeds:
            run_training(cfg, seed)
            rows.append({"kind": kind, "seed": seed})
    pd.DataFrame(rows).to_csv("runs/baselines_index.csv", index=False)
