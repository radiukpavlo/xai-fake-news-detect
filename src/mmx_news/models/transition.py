from __future__ import annotations

import numpy as np
from numpy.linalg import svd


def fit_transition(A: np.ndarray, B: np.ndarray, tol: float = 1.0e-8) -> np.ndarray:
    """Compute T = A^+ B via SVD with tolerance.

    Parameters
    ----------
    A : np.ndarray (m, k)
    B : np.ndarray (m, ell)
    tol : float
        Relative threshold for singular values.

    Returns
    -------
    np.ndarray
        Transition matrix T of shape (k, ell).
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays")
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have same number of rows")
    U, s, Vt = svd(A, full_matrices=False)
    if s.size == 0:
        raise ValueError("Singular values empty; A likely empty")
    max_s = np.max(s)
    s_inv = np.array([1.0 / si if (si / max_s) > tol else 0.0 for si in s], dtype=A.dtype)
    A_pinv = (Vt.T * s_inv) @ U.T
    return A_pinv @ B
