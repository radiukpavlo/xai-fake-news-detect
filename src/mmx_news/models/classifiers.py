from __future__ import annotations

import numpy as np
from sklearn import svm, linear_model


def make_classifier(kind: str = "linear_svm", C: float = 1.0, gamma: float = 0.5):
    kind = str(kind).lower()
    if kind == "linear_svm":
        return svm.LinearSVC(C=C)
    if kind == "rbf_svm":
        return svm.SVC(C=C, kernel="rbf", gamma=gamma, probability=False)
    if kind == "logreg":
        return linear_model.LogisticRegression(C=C, max_iter=1000, n_jobs=1)
    raise ValueError(f"Unsupported classifier type: {kind}")
