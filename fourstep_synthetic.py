"""
fourstep_synthetic.py

Synthetic four-step travel demand model for a 20-TAZ city.
Stage 1: classical model on synthetic data (no AI yet).

Author: (Your Name)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
import networkx as nx

# -------------------------------------------------
# GLOBAL SETTINGS
# -------------------------------------------------

RANDOM_SEED = 42
NUM_ZONES = 20

rng = np.random.default_rng(RANDOM_SEED)

# -------------------------------------------------
# 1. SYNTHETIC CITY GENERATOR (TAZ-LEVEL DATA)
# -------------------------------------------------

@dataclass
class SyntheticCity:
    taz: pd.DataFrame                 # zone attributes
    distance_matrix: pd.DataFrame     # minutes between TAZs (symmetric)
    travel_time_matrix: pd.DataFrame  # base car travel time (minutes)


def generate_synthetic_city(num_zones: int = NUM_ZONES,
                            seed: int = RANDOM_SEED) -> SyntheticCity:
    """
    Generate synthetic socio-economic and spatial data for a set of TAZs.

    Returns
    -------
    SyntheticCity
    """
    rng_local = np.random.default_rng(seed)

    # Create synthetic 2D coordinates for zones (km), roughly a 10x10 km city
    x = rng_local.uniform(0, 10, size=num_zones)
    y = rng_local.uniform(0, 10, size=num_zones)

    # Population and households
    population = rng_local.normal(loc=25000, scale=5000, size=num_zones)
    population = np.clip(population, 8000, None).astype(int)

    households = (population / rng_local.normal(loc=3.2, scale=0.3,
                                                size=num_zones)).astype(int)

    # Workers and students
    workers = (population * rng_local.uniform(0.35, 0.45, size=num_zones)).astype(int)
    students = (population * rng_local.uniform(0.2, 0.3, size=num_zones)).astype(int)

    # Income (monthly, arbitrary units) – lognormal
    income = rng_local.lognormal(mean=10, sigma=0.4, size=num_zones)

    # Car ownership rate as sigmoid of income
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    car_ownership_rate = sigmoid(0.00003 * income - 3.0)
    cars = (car_ownership_rate * households * rng_local.uniform(0.8, 1.2,
                                                                size=num_zones)).astype(int)

    # Land-use mix index (0–1)
    land_use_mix = rng_local.uniform(0.2, 0.9, size=num_zones)

    # Jobs and floor areas
    service_jobs = (workers * rng_local.uniform(0.8, 1.4, size=num_zones)).astype(int)
    industrial_jobs = (workers * rng_local.uniform(0.3, 0.8, size=num_zones)).astype(int)
    retail_jobs = (workers * rng_local.uniform(0.3, 0.7, size=num_zones)).astype(int)

    school_capacity = (students * rng_local.uniform(1.1, 1.5, size=num_zones)).astype(int)
    retail_floor_area = (retail_jobs * rng_local.uniform(20, 40, size=num_zones))  # arbitrary units

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
    })

    taz_df.set_index("TAZ", inplace=True)

    # Distance matrix (Euclidean) and base car travel time (min)
    coords = taz_df[["x_km", "y_km"]].to_numpy()
    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist_km = np.sqrt(dx ** 2 + dy ** 2)

    # Assume average car speed ~ 25–35 km/h plus 3–8 minutes terminal time
    avg_speed_kmh = rng_local.uniform(25, 35)
    tt_base = (dist_km / avg_speed_kmh) * 60  # minutes
    tt_matrix = tt_base + rng_local.uniform(3, 8, size=(num_zones, num_zones))

    # Ensure diagonal is small (intra-zonal trips)
    np.fill_diagonal(tt_matrix, rng_local.uniform(3, 5, size=num_zones))
    np.fill_diagonal(dist_km, rng_local.uniform(0.2, 0.5, size=num_zones))

    distance_df = pd.DataFrame(dist_km,
                               index=taz_df.index,
                               columns=taz_df.index)
    tt_df = pd.DataFrame(tt_matrix,
                         index=taz_df.index,
                         columns=taz_df.index)

    return SyntheticCity(taz=taz_df,
                         distance_matrix=distance_df,
                         travel_time_matrix=tt_df)

# -------------------------------------------------
# 2. TRIP GENERATION
# -------------------------------------------------

PURPOSES = ["HBW", "HBE", "HBS"]


def trip_generation(taz: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic trip productions and attractions by purpose.

    Parameters
    ----------
    taz : DataFrame
        TAZ-level socio-economic attributes.

    Returns
    -------
    productions : DataFrame (index=TAZ, columns=PURPOSES)
    attractions : DataFrame (index=TAZ, columns=PURPOSES)
    """
    df = taz

    # Productions (synthetic "true" equations)
    P_HBW = 0.8 * df["workers"] + 0.2 * df["cars"]
    P_HBE = 1.2 * df["students"]
    P_HBS = 0.4 * df["households"]

    productions = pd.DataFrame({
        "HBW": P_HBW,
        "HBE": P_HBE,
        "HBS": P_HBS
    }, index=df.index)

    # Attractions (jobs, schools, retail)
    A_HBW = 0.7 * df["service_jobs"] + 0.3 * df["industrial_jobs"]
    A_HBE = 1.5 * df["school_capacity"]
    A_HBS = 1.3 * df["retail_floor_area"]

    attractions = pd.DataFrame({
        "HBW": A_HBW,
        "HBE": A_HBE,
        "HBS": A_HBS
    }, index=df.index)

    # Balance productions and attractions for each purpose
    for p in PURPOSES:
        total_P = productions[p].sum()
        total_A = attractions[p].sum()
        if total_A <= 0:
            continue
        factor = total_P / total_A
        attractions[p] *= factor

    return productions, attractions

# -------------------------------------------------
# 3. GRAVITY-BASED TRIP DISTRIBUTION WITH IPF
# -------------------------------------------------

def gravity_impedance(travel_time_min: np.ndarray,
                      beta: float = 1.5) -> np.ndarray:
    """
    Simple impedance function f(c_ij) = c_ij^beta.

    Smaller f => more attractive; will be inverted later.
    """
    c = np.maximum(travel_time_min, 1e-3)
    return c ** beta


def gravity_distribution(productions: pd.Series,
                         attractions: pd.Series,
                         travel_time: pd.DataFrame,
                         beta: float = 1.5,
                         max_iter: int = 1000,
                         tol: float = 1e-4) -> pd.DataFrame:
    """
    Gravity model with iterative proportional fitting (IPF) to match
    row and column totals.

    Parameters
    ----------
    productions : Series
    attractions : Series
    travel_time : DataFrame
    beta : float
    max_iter : int
    tol : float

    Returns
    -------
    T : DataFrame (OD matrix)
    """
    zones = productions.index
    c = travel_time.loc[zones, zones].to_numpy()
    f = gravity_impedance(c, beta=beta)

    P = productions.to_numpy()
    A = attractions.to_numpy()

    # Initial unbalanced matrix
    W = np.outer(P, A) / f
    W[W < 0] = 0.0

    T = W.copy()
    # IPF
    for _ in range(max_iter):
        # Row adjustment
        row_sums = T.sum(axis=1)
        row_factors = np.divide(P, row_sums,
                                out=np.ones_like(P),
                                where=row_sums > 0)
        T = (T.T * row_factors).T

        # Column adjustment
        col_sums = T.sum(axis=0)
        col_factors = np.divide(A, col_sums,
                                out=np.ones_like(A),
                                where=col_sums > 0)
        T = T * col_factors

        # Convergence check
        row_err = np.abs(T.sum(axis=1) - P).sum()
        col_err = np.abs(T.sum(axis=0) - A).sum()
        if row_err < tol and col_err < tol:
            break

    T_df = pd.DataFrame(T, index=zones, columns=zones)
    return T_df


def build_all_od_matrices(productions: pd.DataFrame,
                          attractions: pd.DataFrame,
                          travel_time: pd.DataFrame,
                          beta_by_purpose: Dict[str, float] | None = None
                          ) -> Dict[str, pd.DataFrame]:
    """
    Build OD matrices for each purpose.

    Returns
    -------
    od_mats : dict[purpose -> DataFrame]
    """
    if beta_by_purpose is None:
        beta_by_purpose = {"HBW": 1.5, "HBE": 1.6, "HBS": 1.4}

    od_mats = {}
    for p in PURPOSES:
        od_mats[p] = gravity_distribution(
            productions[p], attractions[p],
            travel_time=travel_time,
            beta=beta_by_purpose.get(p, 1.5),
        )
    return od_mats

# -------------------------------------------------
# 4. MODE CHOICE (MULTINOMIAL LOGIT)
# -------------------------------------------------

MODES = ["car", "metro", "bus"]


@dataclass
class ModeChoiceResult:
    probabilities: Dict[str, pd.DataFrame]   # mode -> P_ij
    volumes: Dict[str, pd.DataFrame]         # mode -> T_ij^mode
    total_od: pd.DataFrame                   # aggregate OD (all purposes)


def synthetic_mode_choice_costs(travel_time_car: pd.DataFrame
                                ) -> Tuple[Dict[str, pd.DataFrame],
                                           Dict[str, pd.DataFrame]]:
    """
    Given base car travel time, build synthetic time and cost matrices
    for each mode.

    Returns
    -------
    time_mats : dict[mode -> DataFrame]
    cost_mats : dict[mode -> DataFrame]
    """
    tt_car = travel_time_car.copy()
    zones = tt_car.index

    # Metro is faster, bus is slower
    tt_metro = tt_car * 0.8
    tt_bus = tt_car * 1.3

    # Costs (arbitrary synthetic)
    dist_factor = tt_car / 60 * 30  # ~ distance proxy (km)
    cost_car = 2 + 0.12 * dist_factor  # fuel + parking etc.
    cost_metro = 15 + 0.02 * dist_factor  # base fare + distance
    cost_bus = 8 + 0.03 * dist_factor

    time_mats = {
        "car": tt_car,
        "metro": tt_metro,
        "bus": tt_bus
    }
    cost_mats = {
        "car": cost_car,
        "metro": cost_metro,
        "bus": cost_bus
    }
    return time_mats, cost_mats


def mode_choice(od_mats: Dict[str, pd.DataFrame],
                taz: pd.DataFrame,
                travel_time_car: pd.DataFrame,
                beta_time: float = -0.06,
                beta_cost: float = -0.03,
                beta_car_own: float = 0.5
                ) -> ModeChoiceResult:
    """
    Multinomial logit mode choice applied to aggregate OD flows
    (sum over purposes).

    Parameters
    ----------
    od_mats : dict[purpose -> OD matrix]
    taz : DataFrame
    travel_time_car : DataFrame

    Returns
    -------
    ModeChoiceResult
    """
    zones = travel_time_car.index
    # Aggregate OD across purposes
    total_od = sum(od_mats.values())
    total_od = total_od.loc[zones, zones]

    time_mats, cost_mats = synthetic_mode_choice_costs(travel_time_car)

    # Car ownership by origin
    car_own = taz["car_ownership_rate"].reindex(zones).to_numpy()

    n = len(zones)
    car_own_matrix = np.repeat(car_own[:, None], n, axis=1)

    utilities = {}
    for mode in MODES:
        tt = time_mats[mode].to_numpy()
        cost = cost_mats[mode].to_numpy()

        if mode == "car":
            U = beta_time * tt + beta_cost * cost + beta_car_own * car_own_matrix
        else:
            U = beta_time * tt + beta_cost * cost
        utilities[mode] = U

    # Compute probabilities
    exp_U_sum = np.zeros_like(next(iter(utilities.values())))
    for U in utilities.values():
        exp_U_sum += np.exp(U)

    probabilities = {}
    for mode, U in utilities.items():
        P = np.exp(U) / np.maximum(exp_U_sum, 1e-12)
        probabilities[mode] = pd.DataFrame(P, index=zones, columns=zones)

    # Mode-specific flows
    volumes = {}
    total_od_np = total_od.to_numpy()
    for mode in MODES:
        volumes[mode] = pd.DataFrame(
            total_od_np * probabilities[mode].to_numpy(),
            index=zones, columns=zones
        )

    return ModeChoiceResult(
        probabilities=probabilities,
        volumes=volumes,
        total_od=total_od
    )

# -------------------------------------------------
# 5. SYNTHETIC NETWORK & AON ROUTE ASSIGNMENT
# -------------------------------------------------

@dataclass
class Network:
    G: nx.DiGraph
    link_df: pd.DataFrame           # index: link id, columns: from, to, ff_time, capacity, distance
    taz_to_node: Dict[int, int]     # mapping from TAZ -> nearest node


def generate_synthetic_network(taz: pd.DataFrame,
                               avg_speed_kmh: float = 30.0,
                               seed: int = RANDOM_SEED) -> Network:
    """
    Build a synthetic directed network using TAZ centroids plus extra connectors.

    Strategy:
    - Use TAZ centroids as main nodes.
    - Connect each node to its k nearest neighbours (k=3) both directions.

    Returns
    -------
    Network
    """
    rng_local = np.random.default_rng(seed)
    coords = taz[["x_km", "y_km"]].to_numpy()
    zones = taz.index.to_list()
    n = len(zones)

    G = nx.DiGraph()
    for i, z in enumerate(zones):
        G.add_node(z, x=coords[i, 0], y=coords[i, 1])

    # Connect to k nearest neighbours
    k = 3
    link_records = []
    link_id = 0

    for i, zi in enumerate(zones):
        xi, yi = coords[i]
        # distances to others
        dx = coords[:, 0] - xi
        dy = coords[:, 1] - yi
        dist = np.sqrt(dx ** 2 + dy ** 2)
        order = np.argsort(dist)
        # take nearest k excluding itself
        neighbours_idx = [j for j in order if j != i][:k]
        for j in neighbours_idx:
            zj = zones[j]
            d_km = dist[j]
            if d_km <= 0:
                continue
            ff_time = (d_km / avg_speed_kmh) * 60  # minutes
            # capacity (veh/h) synthetic
            cap = rng_local.integers(1200, 2400)

            G.add_edge(zi, zj, length_km=d_km, ff_time=ff_time, capacity=cap)

            link_records.append({
                "link_id": link_id,
                "from": zi,
                "to": zj,
                "distance_km": d_km,
                "ff_time_min": ff_time,
                "capacity_vehph": cap
            })
            link_id += 1

    link_df = pd.DataFrame(link_records).set_index("link_id")

    # Map each TAZ directly to its node (here they coincide)
    taz_to_node = {int(z): int(z) for z in zones}

    return Network(G=G, link_df=link_df, taz_to_node=taz_to_node)


def aon_assignment(od_matrix: pd.DataFrame,
                   network: Network) -> pd.DataFrame:
    """
    All-or-nothing assignment of OD matrix to network links
    using free-flow travel time as cost.

    Parameters
    ----------
    od_matrix : DataFrame (TAZ x TAZ)
    network : Network

    Returns
    -------
    link_flows : DataFrame (index=link_id, column='flow')
    """
    G = network.G
    taz_to_node = network.taz_to_node
    zones = od_matrix.index.to_list()
    flows = np.zeros(len(network.link_df), dtype=float)

    # Precompute a mapping from (u,v) to link_id
    edge_to_link = {}
    for lid, row in network.link_df.iterrows():
        edge_to_link[(row["from"], row["to"])] = lid

    # Use ff_time as edge weight
    for (u, v, data) in G.edges(data=True):
        if "ff_time" not in data:
            data["ff_time"] = data.get("ff_time_min", 1.0)

    # For each OD pair, find shortest path and add flow
    for i, o in enumerate(zones):
        origin_node = taz_to_node[int(o)]
        for j, d in enumerate(zones):
            if i == j:
                continue
            dest_node = taz_to_node[int(d)]
            demand = od_matrix.iat[i, j]
            if demand <= 0:
                continue
            try:
                path = nx.shortest_path(G, origin_node, dest_node,
                                        weight="ff_time")
            except nx.NetworkXNoPath:
                continue
            # accumulate flow on each edge of path
            for k in range(len(path) - 1):
                u = path[k]
                v = path[k + 1]
                lid = edge_to_link.get((u, v))
                if lid is not None:
                    flows[lid] += demand

    link_flows = network.link_df.copy()
    link_flows["flow_vehph"] = flows
    return link_flows

# -------------------------------------------------
# 6. QUICK DEMO (RUN THIS FILE DIRECTLY)
# -------------------------------------------------

if __name__ == "__main__":
    # 1. Generate synthetic city
    city = generate_synthetic_city(num_zones=NUM_ZONES)
    taz = city.taz
    print("TAZ sample:\n", taz.head(), "\n")

    # 2. Trip generation
    productions, attractions = trip_generation(taz)
    print("Total productions by purpose:\n", productions.sum(), "\n")
    print("Total attractions by purpose:\n", attractions.sum(), "\n")

    # 3. OD matrices by gravity
    od_mats = build_all_od_matrices(productions, attractions,
                                    travel_time=city.travel_time_matrix)
    for p, od in od_mats.items():
        print(f"OD matrix ({p}) total trips: {od.values.sum():.1f}")

    # 4. Mode choice
    mc_result = mode_choice(od_mats, taz, city.travel_time_matrix)
    print("\nMode shares (total trips):")
    total_trips = mc_result.total_od.values.sum()
    for m in MODES:
        trips_m = mc_result.volumes[m].values.sum()
        print(f"  {m}: {trips_m:.1f} ({100 * trips_m / total_trips:.1f} %)")

    # 5. Network & AON assignment (using car OD only as example)
    network = generate_synthetic_network(taz)
    car_od = mc_result.volumes["car"]
    link_flows = aon_assignment(car_od, network)
    print("\nLink flows (first 10):\n", link_flows.head(10))
