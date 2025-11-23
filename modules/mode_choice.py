# modules/mode_choice.py
# TripAI – Multinomial Logit Mode Choice

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class ModeChoiceResult:
    """
    Container for mode choice outputs.

    Attributes
    ----------
    total_od : pd.DataFrame
        OD matrix summed over all purposes.
    volumes : Dict[str, pd.DataFrame]
        Mode-specific OD matrices (Car/Metro/Bus).
    probabilities : Dict[str, pd.DataFrame]
        Mode choice probabilities per OD pair.
    """
    total_od: pd.DataFrame
    volumes: Dict[str, pd.DataFrame]
    probabilities: Dict[str, pd.DataFrame]


def mode_choice(
    od_matrices: Dict[str, pd.DataFrame],
    taz: pd.DataFrame,
    travel_time: pd.DataFrame,
    beta_time: float = -0.06,
    beta_cost: float = -0.03,
    beta_car_own: float = 0.5,
) -> ModeChoiceResult:
    """
    Apply a simple Multinomial Logit (MNL) mode choice model for
    Car / Metro / Bus.

    U_m = β_time * time_m + β_cost * cost_m + γ * car_ownership (for car only)

    Parameters
    ----------
    od_matrices : dict[str, pd.DataFrame]
        OD matrices by purpose (HBW, HBE, HBS).
    taz : pd.DataFrame
        TAZ attributes with 'car_ownership_rate', 'x_km', 'y_km'.
    travel_time : pd.DataFrame
        Base car travel time matrix (minutes).
    beta_time : float
        Coefficient on in-vehicle travel time.
    beta_cost : float
        Coefficient on generalized cost.
    beta_car_own : float
        Additional utility for Car associated with car ownership rate.

    Returns
    -------
    ModeChoiceResult
        Aggregated OD, volumes by mode, and probabilities by mode.
    """
    zones = travel_time.index

    # 1. Aggregate OD across purposes
    total_od = None
    for mat in od_matrices.values():
        if total_od is None:
            total_od = mat.copy()
        else:
            total_od += mat
    total_od = total_od.loc[zones, zones]

    # 2. Build time and cost matrices for each mode
    tt_car = travel_time.loc[zones, zones].astype(float)
    tt_metro = tt_car * 0.8  # metro faster
    tt_bus = tt_car * 1.3    # bus slower

    # Distance proxy (km)
    dist_proxy = tt_car / 60.0 * 30.0  # 30 km/h

    cost_car = 2.0 + 0.12 * dist_proxy
    cost_metro = 15.0
    cost_bus = 8.0 + 0.03 * dist_proxy

    # 3. Car ownership matrix
    car_own = taz["car_ownership_rate"].reindex(zones).to_numpy()
    n = len(zones)
    car_own_matrix = np.repeat(car_own[:, None], n, axis=1)

    # 4. Utilities
    modes = ["car", "metro", "bus"]
    utilities = {}

    # Car
    U_car = (
        beta_time * tt_car.to_numpy()
        + beta_cost * cost_car.to_numpy()
        + beta_car_own * car_own_matrix
    )
    utilities["car"] = U_car

    # Metro
    U_metro = beta_time * tt_metro.to_numpy() + beta_cost * cost_metro.to_numpy()
    utilities["metro"] = U_metro

    # Bus
    U_bus = beta_time * tt_bus.to_numpy() + beta_cost * cost_bus.to_numpy()
    utilities["bus"] = U_bus

    # 5. Probabilities via softmax
    exp_sum = np.zeros_like(U_car)
    for U in utilities.values():
        exp_sum += np.exp(U)

    probabilities: Dict[str, pd.DataFrame] = {}
    for mode, U in utilities.items():
        P = np.exp(U) / np.maximum(exp_sum, 1e-12)
        probabilities[mode] = pd.DataFrame(P, index=zones, columns=zones)

    # 6. Mode-specific OD flows
    volumes: Dict[str, pd.DataFrame] = {}
    total_np = total_od.to_numpy()
    for mode in modes:
        volumes[mode] = pd.DataFrame(
            total_np * probabilities[mode].to_numpy(),
            index=zones,
            columns=zones,
        )

    return ModeChoiceResult(
        total_od=total_od,
        volumes=volumes,
        probabilities=probabilities,
    )
