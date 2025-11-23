# modules/route_assignment.py
# TripAI – Synthetic Network + AON + Frank–Wolfe UE

from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Network:
    """
    Simple synthetic network representation where each TAZ pair
    (i, j) is connected by a single directed link.

    Attributes
    ----------
    links : pd.DataFrame
        Columns:
        - link_id
        - from_zone
        - to_zone
        - length_km
        - t0_min        (free-flow travel time)
        - capacity_vehph
        - alpha         (BPR parameter)
        - beta          (BPR parameter)
    """
    links: pd.DataFrame


def generate_synthetic_network(taz: pd.DataFrame) -> Network:
    """
    Generate a fully connected directed network over TAZ centroids.
    Each ordered pair (i, j), i != j, is represented as a distinct link.

    Travel times are approximated from Euclidean distance and an
    assumed free-flow speed.

    Parameters
    ----------
    taz : pd.DataFrame
        Must include 'x_km' and 'y_km' columns.

    Returns
    -------
    Network
    """
    zones = taz.index.to_list()
    coords = taz[["x_km", "y_km"]].to_numpy()
    n = len(zones)

    rows = []
    link_id = 0
    ff_speed_kmh = 30.0

    for i_idx, i in enumerate(zones):
        for j_idx, j in enumerate(zones):
            if i == j:
                continue
            dx = coords[j_idx, 0] - coords[i_idx, 0]
            dy = coords[j_idx, 1] - coords[i_idx, 1]
            dist = np.sqrt(dx**2 + dy**2)  # km
            t0 = (dist / max(ff_speed_kmh, 1e-3)) * 60.0 + 3.0  # minutes

            rows.append(
                {
                    "link_id": link_id,
                    "from_zone": i,
                    "to_zone": j,
                    "length_km": dist,
                    "t0_min": t0,
                    "capacity_vehph": np.random.uniform(1500, 2500),
                    "alpha": 0.15,
                    "beta": 4.0,
                }
            )
            link_id += 1

    links_df = pd.DataFrame(rows).set_index("link_id")
    return Network(links=links_df)


def _init_flow_column(links: pd.DataFrame, col: str = "flow_vehph") -> pd.DataFrame:
    df = links.copy()
    if col not in df.columns:
        df[col] = 0.0
    else:
        df[col] = 0.0
    return df


def aon_assignment(od_car: pd.DataFrame, network: Network) -> pd.DataFrame:
    """
    All-or-nothing (AON) assignment assuming a single direct link
    between each TAZ pair (i, j). All demand from i to j is loaded
    on that link.

    Parameters
    ----------
    od_car : pd.DataFrame
        Car OD matrix (veh/h equivalent).
    network : Network

    Returns
    -------
    pd.DataFrame
        Link flows with column 'flow_vehph'.
    """
    links = _init_flow_column(network.links, col="flow_vehph")
    zones = od_car.index.to_list()

    for i in zones:
        for j in zones:
            if i == j:
                continue
            q = float(od_car.loc[i, j])
            if q <= 0:
                continue
            mask = (links["from_zone"] == i) & (links["to_zone"] == j)
            links.loc[mask, "flow_vehph"] += q

    return links


def _bpr_travel_time(
    flows: np.ndarray,
    t0: np.ndarray,
    capacity: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Standard BPR volume-delay function."""
    vc = np.divide(flows, capacity, out=np.zeros_like(flows), where=capacity > 0)
    return t0 * (1.0 + alpha * np.power(vc, beta))


def frank_wolfe_ue(
    od_car: pd.DataFrame,
    network: Network,
    max_iter: int = 30,
) -> pd.DataFrame:
    """
    Very simple Frank–Wolfe style User Equilibrium assignment over
    the synthetic network where each OD pair has a single link.

    Because there is only one 'route' per OD, the UE solution
    coincides with the AON solution. This implementation still
    outlines the iterative structure for pedagogical purposes.

    Parameters
    ----------
    od_car : pd.DataFrame
        Car OD matrix (veh/h).
    network : Network
    max_iter : int
        Maximum iterations (for demonstration).

    Returns
    -------
    pd.DataFrame
        Link flows with column 'flow_vehph' and implied travel times.
    """
    links = network.links.copy()
    n_links = len(links)

    # Initialize flows
    flows = np.zeros(n_links, dtype=float)

    # Extract BPR parameters
    t0 = links["t0_min"].to_numpy()
    cap = links["capacity_vehph"].to_numpy()
    alpha = links["alpha"].to_numpy()
    beta = links["beta"].to_numpy()

    # Pre-build a mapping (from_zone, to_zone) -> link indices
    index = links.reset_index()
    zone_pairs = {}
    for idx, row in index.iterrows():
        key = (row["from_zone"], row["to_zone"])
        zone_pairs[key] = row["link_id"]

    # Iterate Frank–Wolfe (though it converges immediately in this simple network)
    for k in range(max_iter):
        # Step 1: Compute travel times (not used for path choice here)
        tt = _bpr_travel_time(flows, t0, cap, alpha, beta)

        # Step 2: AON step (all or nothing given current times – here trivial)
        aon_flows = np.zeros_like(flows)
        zones = od_car.index.to_list()
        for i in zones:
            for j in zones:
                if i == j:
                    continue
                q = float(od_car.loc[i, j])
                if q <= 0:
                    continue
                lid = zone_pairs[(i, j)]
                aon_flows[lid] += q

        # Step 3: Line search step-size (generic diminishing rule)
        step = 2.0 / (k + 2.0)
        new_flows = flows + step * (aon_flows - flows)

        # Convergence check
        if np.allclose(new_flows, flows, atol=1e-3):
            flows = new_flows
            break

        flows = new_flows

    links["flow_vehph"] = flows
    links["tt_min"] = _bpr_travel_time(flows, t0, cap, alpha, beta)
    return links
