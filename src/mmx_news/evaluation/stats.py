from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import ttest_rel


def paired_t_test(a: Iterable[float], b: Iterable[float]) -> Tuple[float, float]:
    """Return (t_stat, p_value) for paired two-sided t-test."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    t, p = ttest_rel(a, b)
    return float(t), float(p)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni correction decisions for hypotheses corresponding to p_values."""
    m = len(p_values)
    idx = np.argsort(p_values)
    decisions = [False] * m
    for k, i in enumerate(idx):
        threshold = alpha / (m - k)
        if p_values[i] <= threshold:
            decisions[i] = True
        else:
            break
    return decisions
