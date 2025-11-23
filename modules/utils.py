# modules/utils.py
# TripAI – Utility functions (IPF, etc.)

from __future__ import annotations
import numpy as np
import pandas as pd


def iterative_proportional_fitting(
    T_init: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Perform Iterative Proportional Fitting (IPF) to adjust an initial
    non-negative matrix T_init so that its row and column sums match
    given marginals.

    Parameters
    ----------
    T_init : np.ndarray
        Initial non-negative matrix (NxN).
    row_targets : np.ndarray
        Target row sums (length N).
    col_targets : np.ndarray
        Target column sums (length N).
    max_iter : int
        Maximum number of IPF iterations.
    tol : float
        Convergence tolerance on row/column marginal differences.

    Returns
    -------
    np.ndarray
        Adjusted matrix with approximately matching row/column totals.
    """
    T = T_init.copy().astype(float)
    n = T.shape[0]

    for _ in range(max_iter):
        # Row scaling
        row_sums = T.sum(axis=1)
        row_factors = np.ones(n)
        mask = row_sums > 0
        row_factors[mask] = row_targets[mask] / row_sums[mask]
        T *= row_factors[:, None]

        # Column scaling
        col_sums = T.sum(axis=0)
        col_factors = np.ones(n)
        mask = col_sums > 0
        col_factors[mask] = col_targets[mask] / col_sums[mask]
        T *= col_factors[None, :]

        # Check convergence
        if (
            np.allclose(T.sum(axis=1), row_targets, atol=tol)
            and np.allclose(T.sum(axis=0), col_targets, atol=tol)
        ):
            break

    return T
