# modules/ai_link_flow_emulator.py
# TripAI – AI Emulator for Link Flows

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .route_assignment import aon_assignment, Network


@dataclass
class LinkFlowEmulator:
    """
    AI emulator for link flows under scaled OD demand.

    Attributes
    ----------
    model : RandomForestRegressor
        Multi-output regressor mapping demand scale -> link flows.
    link_ids : np.ndarray
        IDs of links in the same order as training outputs.
    base_total_demand : float
        Total baseline car OD (for reference).
    """
    model: RandomForestRegressor
    link_ids: np.ndarray
    base_total_demand: float


def _generate_training_scenarios(
    base_od: np.ndarray,
    network: Network,
    n_scenarios: int = 20,
    low_scale: float = 0.7,
    high_scale: float = 1.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training scenarios by scaling baseline OD and performing
    AON assignment to obtain link flows.

    Parameters
    ----------
    base_od : np.ndarray
        Baseline OD matrix (veh/h).
    network : Network
    n_scenarios : int
        Number of random scaling scenarios.
    low_scale : float
        Minimum demand scale factor.
    high_scale : float
        Maximum demand scale factor.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_scenarios, 1) – the demand scale.
    Y : np.ndarray
        Target matrix of shape (n_scenarios, n_links) – link flows.
    """
    n_zones = base_od.shape[0]
    n_links = len(network.links)
    scales = np.random.uniform(low_scale, high_scale, size=n_scenarios)

    X = scales.reshape(-1, 1)
    Y = np.zeros((n_scenarios, n_links), dtype=float)

    # Build index -> (from_zone, to_zone) map to reuse AON logic
    # We will call the existing aon_assignment with scaled OD each time.
    # Convert base OD to DataFrame with synthetic zone index 0..n-1.
    zones = np.arange(n_zones)
    base_od_df = pd.DataFrame(base_od, index=zones, columns=zones)

    for idx, s in enumerate(scales):
        od_scaled = base_od_df * s
        flows_df = aon_assignment(od_scaled, network)
        Y[idx, :] = flows_df["flow_vehph"].to_numpy()

    return X, Y


def train_link_flow_emulator(
    base_car_od: np.ndarray,
    network: Network,
    n_scenarios: int = 20,
) -> tuple[LinkFlowEmulator, pd.DataFrame]:
    """
    Train a simple RandomForest-based emulator that maps a single
    scalar 'demand scale' to resulting link flows.

    Parameters
    ----------
    base_car_od : np.ndarray
        Baseline car OD matrix (veh/h).
    network : Network
    n_scenarios : int
        Number of training scenarios.

    Returns
    -------
    emulator : LinkFlowEmulator
    training_history : pd.DataFrame
        Scenario scales and corresponding total flows, for diagnostics.
    """
    X, Y = _generate_training_scenarios(base_car_od, network, n_scenarios=n_scenarios)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
    )
    model.fit(X, Y)

    link_ids = network.links.index.to_numpy()
    base_total = float(base_car_od.sum())

    # Build training history for inspection
    total_flows = Y.sum(axis=1)
    training_history = pd.DataFrame(
        {
            "scale": X.flatten(),
            "total_link_flow": total_flows,
        }
    )

    emulator = LinkFlowEmulator(
        model=model,
        link_ids=link_ids,
        base_total_demand=base_total,
    )
    return emulator, training_history


def predict_link_flows(
    emulator: LinkFlowEmulator,
    scale: float,
    network: Network,
) -> pd.DataFrame:
    """
    Predict link flows for a new demand scale using the trained emulator.

    Parameters
    ----------
    emulator : LinkFlowEmulator
    scale : float
        Multiplicative scaling factor relative to baseline OD.
    network : Network

    Returns
    -------
    pd.DataFrame
        Link table with predicted flows in column 'flow_vehph_emulated'.
    """
    X_new = np.array([[scale]], dtype=float)
    y_pred = emulator.model.predict(X_new).flatten()

    links = network.links.copy()
    links["flow_vehph_emulated"] = y_pred
    return links
