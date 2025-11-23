# modules/synthetic_city.py
# TripAI – Synthetic City Generator (20 TAZ by default)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# Global defaults
RANDOM_SEED = 42
NUM_ZONES_DEFAULT = 20


@dataclass
class SyntheticCity:
    """
    Container for synthetic city data.
    """
    taz: pd.DataFrame                 # TAZ-level attributes
    distance_matrix: pd.DataFrame     # inter-TAZ distances (km)
    travel_time_matrix: pd.DataFrame  # inter-TAZ travel times (minutes)


def generate_synthetic_city(
    num_zones: int = NUM_ZONES_DEFAULT,
    seed: Optional[int] = RANDOM_SEED
) -> SyntheticCity:
    """
    Generate a synthetic metropolitan region with a specified number of
    Traffic Analysis Zones (TAZs).

    Outputs:
    - taz: DataFrame indexed by TAZ id with socio-economic attributes
    - distance_matrix: symmetric TAZ-to-TAZ distances (km)
    - travel_time_matrix: car travel time (minutes)

    Parameters
    ----------
    num_zones : int
        Number of zones (TAZ) to generate. Default = 20.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticCity
    """
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Generate basic spatial layout (coordinates)
    # ---------------------------------------------------------
    # City spread over roughly 10 x 10 km area
    x = rng.uniform(0, 10, size=num_zones)
    y = rng.uniform(0, 10, size=num_zones)

    # ---------------------------------------------------------
    # 2. Socio-economic attributes at TAZ level
    # ---------------------------------------------------------

    # Population distribution (clip to avoid negative / too small)
    population = rng.normal(loc=25000, scale=5000, size=num_zones)
    population = np.clip(population, 8000, None).astype(int)

    # Average household size ~3.2 with small variation
    hh_size = rng.normal(loc=3.2, scale=0.3, size=num_zones)
    households = (population / np.maximum(hh_size, 1.5)).astype(int)

    # Workers and students as shares of population
    workers = (population * rng.uniform(0.35, 0.45, size=num_zones)).astype(int)
    students = (population * rng.uniform(0.20, 0.30, size=num_zones)).astype(int)

    # Monthly income (arbitrary units), lognormal
    income = rng.lognormal(mean=10.0, sigma=0.4, size=num_zones)

    # Car ownership rate as a sigmoid of income
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    car_ownership_rate = sigmoid(0.00003 * income - 3.0)
    cars = (
        car_ownership_rate
        * households
        * rng.uniform(0.8, 1.2, size=num_zones)
    ).astype(int)

    # Land-use mix index (0–1)
    land_use_mix = rng.uniform(0.2, 0.9, size=num_zones)

    # Jobs and floor area
    service_jobs = (workers * rng.uniform(0.8, 1.4, size=num_zones)).astype(int)
    industrial_jobs = (workers * rng.uniform(0.3, 0.8, size=num_zones)).astype(int)
    retail_jobs = (workers * rng.uniform(0.3, 0.7, size=num_zones)).astype(int)

    school_capacity = (students * rng.uniform(1.1, 1.5, size=num_zones)).astype(int)
    retail_floor_area = retail_jobs * rng.uniform(20, 40, size=num_zones)  # arbitrary units

    # Build TAZ DataFrame
    taz_df = pd.DataFrame({
        "TAZ": np.arange(1, num_zones + 1),
        "x_km": x,
        "y_km": y,
        "population": population,
        "households": households,
        "workers": workers,
        "students": students,
        "income": income,
        "car_ownership_rate": car_ownership_rate,
        "cars": cars,
        "land_use_mix": land_use_mix,
        "service_jobs": service_jobs,
        "industrial_jobs": industrial_jobs,
        "retail_jobs": retail_jobs,
        "school_capacity": school_capacity,
        "retail_floor_area": retail_floor_area,
    }).set_index("TAZ")

    # ---------------------------------------------------------
    # 3. Distance & travel time matrices
    # ---------------------------------------------------------
    coords = taz_df[["x_km", "y_km"]].to_numpy()
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist_km = np.sqrt(dx**2 + dy**2)

    # Average car speed (km/h) and base travel times (minutes)
    avg_speed_kmh = rng.uniform(25, 35)
    tt_base = (dist_km / np.maximum(avg_speed_kmh, 1e-3)) * 60.0  # minutes

    # Add random terminal / intersection delays (3–8 minutes)
    tt_matrix = tt_base + rng.uniform(3, 8, size=(num_zones, num_zones))

    # Intra-zonal adjustment (short distances and times)
    np.fill_diagonal(dist_km, rng.uniform(0.2, 0.5, size=num_zones))
    np.fill_diagonal(tt_matrix, rng.uniform(3, 5, size=num_zones))

    distance_df = pd.DataFrame(
        dist_km,
        index=taz_df.index,
        columns=taz_df.index,
    )
    travel_time_df = pd.DataFrame(
        tt_matrix,
        index=taz_df.index,
        columns=taz_df.index,
    )

    # ---------------------------------------------------------
    # 4. Return SyntheticCity object
    # ---------------------------------------------------------
    return SyntheticCity(
        taz=taz_df,
        distance_matrix=distance_df,
        travel_time_matrix=travel_time_df,
    )
