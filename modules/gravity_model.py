# modules/gravity_model.py
# TripAI – Gravity Model + OD Matrix Builder

from __future__ import annotations
import numpy as np
import pandas as pd

from .utils import iterative_proportional_fitting

PURPOSES = ["HBW", "HBE", "HBS"]


def gravity_model_doubly_constrained(
    productions: pd.Series,
    attractions: pd.Series,
    travel_time: pd.DataFrame,
    beta: float = -0.1,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Doubly-constrained gravity model with IPF.

    T_ij ∝ P_i * A_j * f(c_ij),
    where f(c_ij) = exp(beta * c_ij), beta < 0

    IPF is used to ensure row sums match productions and column sums
    match attractions.

    Parameters
    ----------
    productions : pd.Series
        Trip productions P_i by origin TAZ.
    attractions : pd.Series
        Trip attractions A_j by destination TAZ.
    travel_time : pd.DataFrame
        Impedance matrix c_ij (minutes).
    beta : float
        Distance-decay parameter (negative).
    max_iter : int
        Maximum IPF iterations.
    tol : float
        Tolerance for marginal convergence.

    Returns
    -------
    pd.DataFrame
        OD matrix T_ij (index=origins, columns=destinations).
    """
    idx = productions.index
    P = productions.values.astype(float)
    A = attractions.values.astype(float)

    c = travel_time.loc[idx, idx].values.astype(float)
    # Impedance function
    F = np.exp(beta * c)

    # Initial gravity estimate
    T0 = np.outer(P, A) * F
    # Avoid all-zero rows/cols
    T0[T0 < 0] = 0.0

    T_adj = iterative_proportional_fitting(T0, P, A, max_iter=max_iter, tol=tol)

    return pd.DataFrame(T_adj, index=idx, columns=idx)


def build_all_od_matrices(
    productions_df: pd.DataFrame,
    attractions_df: pd.DataFrame,
    travel_time: pd.DataFrame,
    beta: float = -0.1,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, pd.DataFrame]:
    """
    Build OD matrices for all purposes using the doubly-constrained gravity model.

    Parameters
    ----------
    productions_df : pd.DataFrame
        Columns = purposes (HBW, HBE, HBS), index = TAZ.
    attractions_df : pd.DataFrame
        Columns = purposes (HBW, HBE, HBS), index = TAZ.
    travel_time : pd.DataFrame
        Travel time matrix (minutes), index/cols = TAZ.
    beta : float
        Distance-decay parameter for all purposes.
    max_iter : int
        Maximum iterations for IPF.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from purpose -> OD matrix (DataFrame).
    """
    od_mats: dict[str, pd.DataFrame] = {}

    for purpose in PURPOSES:
        P = productions_df[purpose]
        A = attractions_df[purpose]
        od_mats[purpose] = gravity_model_doubly_constrained(
            P, A, travel_time, beta=beta, max_iter=max_iter, tol=tol
        )

    return od_mats
