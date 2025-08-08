from __future__ import annotations

import numpy as np
from mmx_news.evaluation.metrics import binary_metrics
from mmx_news.models.transition import fit_transition


def test_transition_recovery() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 16))
    T_true = rng.normal(size=(16, 5))
    B = A @ T_true
    T_est = fit_transition(A, B)
    rel = np.linalg.norm(A @ T_est - B) / (np.linalg.norm(B) + 1e-12)
    assert rel < 1e-9


def test_binary_metrics_perfect_and_zero() -> None:
    y = np.array([0, 1, 1, 0, 1])
    pred = y.copy()
    m = binary_metrics(y, pred, scores=pred.astype(float))
    assert m.accuracy == 1.0 and m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0

    pred0 = np.zeros_like(y)
    m0 = binary_metrics(y, pred0, scores=pred0.astype(float))
    assert 0.0 <= m0.precision <= 1.0
    assert 0.0 <= m0.recall <= 1.0
    assert 0.0 <= m0.f1 <= 1.0
