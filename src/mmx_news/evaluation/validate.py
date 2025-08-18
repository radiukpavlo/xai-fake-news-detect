from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from .metrics import binary_metrics


def evaluate_checkpoint(checkpoint: str | Path, split: str = "test") -> Dict[str, float]:
    ckpt = Path(checkpoint)
    root = ckpt.parent
    clf = joblib.load(ckpt)
    B = np.load(root / "B.npy")
    splits = np.load(root / "split_indices.npy")
    y = np.load(root / "y.npy")
    if split == "val":
        idx = splits[1]
    else:
        idx = splits[2]
    X = B[idx]
    y_true = y[idx]
    y_pred = clf.predict(X)
    scores = clf.decision_function(X) if hasattr(clf, "decision_function") else None
    m = binary_metrics(y_true, y_pred, scores=scores)
    result = {
        "accuracy": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "auc": (m.auc if isinstance(m.auc, float) else float("nan")),
    }
    with open(root / f"eval_{split}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
