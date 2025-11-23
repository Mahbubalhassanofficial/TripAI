import streamlit as st
import numpy as np
import pandas as pd
import os

from modules.gravity_model import build_all_od_matrices
from modules.route_assignment import generate_synthetic_network, aon_assignment

st.set_page_config(layout="wide")
st.title("⚙️ Policy Scenario Engine")

st.markdown("""
Use this page to test **policy scenarios** on top of the synthetic four-step model:

- 🚈 **Metro Improvement**: faster travel time, lower fare  
- 🚗 **Congestion Charge**: extra cost for car trips into CBD zones  
- 🏙️ **Transit-Oriented Development (TOD)**: increase attractions in selected TAZs  

The engine compares **Baseline vs Scenario** in terms of:
- Mode shares (Car / Metro / Bus)
- Car link flows (AON assignment)
""")

# ------------------------------------------------------
# CHECK REQUIRED STATE
# ------------------------------------------------------
required_keys = ["city", "productions", "attractions", "od", "mode_choice"]
missing = [k for k in required_keys if k not in st.session_state]

if missing:
    st.error(f"Please complete previous steps first. Missing: {', '.join(missing)}")
    st.stop()

city = st.session_state["city"]
taz = city.taz
productions = st.session_state["productions"]
attractions_base = st.session_state["attractions"]
od_base = st.session_state["od"]
mode_choice_base = st.session_state["mode_choice"]
tt_car_base = city.travel_time_matrix.copy()

zones = list(taz.index)

# ------------------------------------------------------
# HELPER: MODE SHARE CALCULATION
# ------------------------------------------------------
def compute_mode_shares(mode_volumes: dict, total_od: pd.DataFrame) -> pd.DataFrame:
    total_trips = total_od.values.sum()
    rows = []
    for m, mat in mode_volumes.items():
        trips = mat.values.sum()
        share = trips / total_trips if total_trips > 0 else 0
        rows.append({"mode": m, "trips": trips, "share": share})
    return pd.DataFrame(rows)

# ------------------------------------------------------
# SIDEBAR – POLICY CONTROLS
# ------------------------------------------------------
st.sidebar.header("Policy Controls")

st.sidebar.subheader("🚈 Metro Improvement")
metro_time_reduction_pct = st.sidebar.slider(
    "Metro travel time reduction (%)", 0, 50, 20
)
metro_fare_change_pct = st.sidebar.slider(
    "Metro fare change (%)", -50, 50, -20
)

st.sidebar.subheader("🚗 Congestion Charge (Car)")
congestion_charge = st.sidebar.slider(
    "Extra generalized cost for car entering CBD", 0.0, 50.0, 20.0, step=1.0
)

default_cbd = zones[:5] if len(zones) >= 5 else zones
cbd_zones = st.sidebar.multiselect(
    "CBD zones (destinations)", options=zones, default=default_cbd
)

st.sidebar.subheader("🏙️ TOD – Modify Attractions")
apply_tod = st.sidebar.checkbox("Apply TOD", value=False)
tod_increase_pct = st.sidebar.slider(
    "Attraction increase (%)", 0, 100, 30
)
tod_zones = st.sidebar.multiselect(
    "TOD zones", options=zones, default=zones[:3] if len(zones) >= 3 else zones
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("▶ Run Scenario")

# ------------------------------------------------------
# BASELINE SUMMARY
# ------------------------------------------------------
st.header("📊 Baseline Summary")

baseline_shares = compute_mode_shares(
    mode_choice_base.volumes, mode_choice_base.total_od
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Baseline Mode Shares")
    st.dataframe(baseline_shares.style.format({"trips": "{:.1f}", "share": "{:.3f}"}))

with col2:
    if "link_flows" in st.session_state:
        st.subheader("Baseline Car Link Flows (sample)")
        st.dataframe(st.session_state["link_flows"].head(10))
    else:
        st.info("Baseline link flows not stored. Run Route Assignment page.")

# ------------------------------------------------------
# BUILD POLICY TIME/COST MATRICES
# ------------------------------------------------------
def build_policy_time_cost_matrices(
        tt_car_base: pd.DataFrame,
        metro_time_reduction_pct: float,
        metro_fare_change_pct: float,
        congestion_charge: float,
        cbd_zones: list
):
    """
    Build modified travel time and cost matrices under policy scenario.
    """
    tt_car = tt_car_base.copy()

    # Base functions: metro faster, bus slower
    tt_metro = tt_car * 0.8 * (1 - metro_time_reduction_pct / 100.0)
    tt_bus = tt_car * 1.3

    # Distance proxy
    dist_proxy = tt_car / 60 * 30

    cost_car = 2 + 0.12 * dist_proxy
    cost_metro = 15 * (1 + metro_fare_change_pct / 100.0)
    cost_bus = 8 + 0.03 * dist_proxy

    # FIX: apply congestion charge vectorized
    cost_car.loc[:, cbd_zones] += congestion_charge

    return (
        {"car": tt_car, "metro": tt_metro, "bus": tt_bus},
        {"car": cost_car, "metro": cost_metro, "bus": cost_bus}
    )

# ------------------------------------------------------
# POLICY MODE CHOICE (Multinomial Logit)
# ------------------------------------------------------
def policy_mode_choice(
        od_mats: dict,
        taz: pd.DataFrame,
        tt_car_base: pd.DataFrame,
        metro_time_reduction_pct: float,
        metro_fare_change_pct: float,
        congestion_charge: float,
        cbd_zones: list,
        beta_time: float = -0.06,
        beta_cost: float = -0.03,
        beta_car_own: float = 0.5
):
    zones = tt_car_base.index

    # Total OD (sum over purposes)
    total_od = sum(od_mats.values())
    total_od = total_od.loc[zones, zones]

    # Updated time/cost
    time_mats, cost_mats = build_policy_time_cost_matrices(
        tt_car_base,
        metro_time_reduction_pct,
        metro_fare_change_pct,
        congestion_charge,
        cbd_zones
    )

    # Car ownership
    car_own = taz["car_ownership_rate"].reindex(zones).to_numpy()
    car_own_matrix = np.repeat(car_own[:, None], len(zones), axis=1)

    modes = ["car", "metro", "bus"]
    utilities = {}

    for mode in modes:
        tt = time_mats[mode].to_numpy()
        cc = cost_mats[mode].to_numpy()

        if mode == "car":
            U = beta_time * tt + beta_cost * cc + beta_car_own * car_own_matrix
        else:
            U = beta_time * tt + beta_cost * cc

        utilities[mode] = U

    # Probabilities
    exp_sum = sum(np.exp(U) for U in utilities.values())
    probabilities = {
        mode: pd.DataFrame(np.exp(U) / np.maximum(exp_sum, 1e-12), index=zones, columns=zones)
        for mode, U in utilities.items()
    }

    # Mode flows
    volumes = {
        mode: pd.DataFrame(
            total_od.to_numpy() * probabilities[mode].to_numpy(),
            index=zones, columns=zones
        )
        for mode in modes
    }

    return probabilities, volumes, total_od

# ------------------------------------------------------
# RUN SCENARIO
# ------------------------------------------------------
if run_button:
    st.header("🧪 Scenario Results")

    # 1) TOD → Modify attractions
    if apply_tod:
        st.subheader("🏙️ TOD Applied — Recomputing OD")

        A_scenario = attractions_base.copy(deep=True)
        factor = 1 + tod_increase_pct / 100.0

        for z in tod_zones:
            if z in A_scenario.index:
                A_scenario.loc[z, ["HBW", "HBS"]] *= factor

        # FIX: consistent call
        od_scenario = build_all_od_matrices(productions, A_scenario, tt_car_base)
    else:
        st.subheader("🏙️ TOD NOT applied — using baseline OD")
        od_scenario = od_base

    # 2) Policy Mode Choice
    st.subheader("🚈 Mode Choice under Policy Scenario")
    probs_scen, vols_scen, total_od_scen = policy_mode_choice(
        od_scenario,
        taz,
        tt_car_base,
        metro_time_reduction_pct,
        metro_fare_change_pct,
        congestion_charge,
        cbd_zones
    )

    # 3) Mode share comparison
    scenario_shares = compute_mode_shares(vols_scen, total_od_scen)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Baseline Mode Shares")
        st.dataframe(baseline_shares.style.format({"trips": "{:.1f}", "share": "{:.3f}"}))
    with colB:
        st.markdown("#### Scenario Mode Shares")
        st.dataframe(scenario_shares.style.format({"trips": "{:.1f}", "share": "{:.3f}"}))

    # 4) AON Car Assignment
    st.subheader("🛣️ Scenario Car Assignment (AON)")

    if "network" in st.session_state:
        network = st.session_state["network"]
    else:
        network = generate_synthetic_network(taz)

    car_od_scen = vols_scen["car"]
    link_flows_scen = aon_assignment(car_od_scen, network)
    st.session_state["link_flows_scenario"] = link_flows_scen

    # FIX: save to data folder
    os.makedirs("data", exist_ok=True)
    link_flows_scen.to_csv("data/link_flows_scenario.csv")

    st.markdown("**Scenario Car Link Flows (sample)**")
    st.dataframe(link_flows_scen.head(10))

    # 5) Summary Numbers
    st.subheader("📉 Key Comparison")

    baseline_car = mode_choice_base.volumes["car"].values.sum()
    scenario_car = vols_scen["car"].values.sum()

    st.write(f"**Baseline car trips:** {baseline_car:,.1f}")
    st.write(f"**Scenario car trips:** {scenario_car:,.1f}")

    if baseline_car > 0:
        pct = 100 * (scenario_car - baseline_car) / baseline_car
        st.write(f"**Change in car trips:** {pct:+.2f}%")

    st.success("Scenario evaluation completed. Use Export page to download results.")

else:
    st.info("Adjust policy parameters on the left, then click **Run Scenario**.")
