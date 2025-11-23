# modules/trip_generation.py
# TripAI – Trip Generation Model

from __future__ import annotations
import pandas as pd
import numpy as np

PURPOSES = ["HBW", "HBE", "HBS"]


def trip_generation(taz: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute trip productions and attractions for each TAZ for three purposes:
    - HBW: Home–Based Work
    - HBE: Home–Based Education
    - HBS: Home–Based Shopping/Other

    The functional forms are deliberately simple but grounded in standard
    trip-rate logic and can be modified for calibration.

    Parameters
    ----------
    taz : pd.DataFrame
        TAZ-level attributes with at least the following columns:
        ['households', 'workers', 'students', 'cars',
         'service_jobs', 'industrial_jobs', 'retail_jobs',
         'school_capacity', 'retail_floor_area'].

    Returns
    -------
    productions : pd.DataFrame
        Index = TAZ, columns = ['HBW', 'HBE', 'HBS'].
    attractions : pd.DataFrame
        Index = TAZ, columns = ['HBW', 'HBE', 'HBS'], balanced so that
        sum(P) = sum(A) for each purpose.
    """
    df = taz.copy()

    # ------------------------------------------------
    # PRODUCTIONS (simple rate-based formulations)
    # ------------------------------------------------
    # HBW: mainly driven by workers and car availability
    P_HBW = 0.8 * df["workers"] + 0.2 * df["cars"]

    # HBE: driven by students
    P_HBE = 1.2 * df["students"]

    # HBS: driven by households (shopping, other)
    P_HBS = 0.4 * df["households"]

    productions = pd.DataFrame(
        {"HBW": P_HBW, "HBE": P_HBE, "HBS": P_HBS},
        index=df.index,
    )

    # ------------------------------------------------
    # ATTRACTIONS (jobs, schools, retail)
    # ------------------------------------------------
    A_HBW = 0.7 * df["service_jobs"] + 0.3 * df["industrial_jobs"]
    A_HBE = 1.5 * df["school_capacity"]
    A_HBS = 1.3 * df["retail_floor_area"]

    attractions = pd.DataFrame(
        {"HBW": A_HBW, "HBE": A_HBE, "HBS": A_HBS},
        index=df.index,
    )

    # ------------------------------------------------
    # SIMPLE BALANCING (one-step scaling)
    # ------------------------------------------------
    for p in PURPOSES:
        total_P = productions[p].sum()
        total_A = attractions[p].sum()
        if total_A > 0:
            attractions[p] *= total_P / total_A

    return productions, attractions
