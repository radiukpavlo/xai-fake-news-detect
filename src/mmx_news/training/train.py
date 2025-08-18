from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .pipeline import (
    build_features_and_embeddings,
    evaluate_model,
    prepare_data,
    save_artifacts,
    train_model,
    _minmax_apply,
)
from .utils import init_run, load_config


def run_training(cfg_path: str, seed: int, mode: str = "full", out_dir: Path | None = None) -> None:
    """Runs the full training pipeline."""
    cfg = load_config(cfg_path)
    run_id, out_dir = init_run(cfg, out_dir=out_dir)

    # 1. Data preparation
    articles, y, idx = prepare_data(cfg, mode=mode)

    # 2. Feature and embedding generation
    A, B_raw, evidences = build_features_and_embeddings(articles, cfg)

    # 3. Model training
    A_train, B_train, y_train = A[idx["train_idx"]], B_raw[idx["train_idx"]], y[idx["train_idx"]]
    T, clf, lo, width = train_model(A_train, B_train, y_train, cfg)

    # 4. Model evaluation
    B_val, y_val = B_raw[idx["val_idx"]], y[idx["val_idx"]]
    B_test, y_test = B_raw[idx["test_idx"]], y[idx["test_idx"]]
    report_val, report_test, auc_val, auc_test = evaluate_model(clf, B_val, y_val, B_test, y_test, lo, width)

    # 5. Artifact persistence
    B = _minmax_apply(B_raw, lo, width)
    Bhat = A @ T
    save_artifacts(
        out_dir=out_dir,
        A=A,
        B=B,
        B_raw=B_raw,
        Bhat=Bhat,
        idx=idx,
        lo=lo,
        width=width,
        evidences=evidences,
        T=T,
        clf=clf,
        report_val=report_val,
        report_test=report_test,
        auc_val=auc_val,
        auc_test=auc_test,
        seed=seed,
        articles=articles,
    )


__all__ = ["run_training"]
