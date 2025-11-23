# pages/11_🎯_Scenario_Optimization.py

import streamlit as st
import numpy as np
import pandas as pd
import os

from typing import Dict

from modules.route_assignment import generate_synthetic_network, frank_wolfe_ue
from modules.ai_link_flow_emulator import predict_link_flows

st.set_page_config(layout="wide")
st.title("🎯 Scenario Optimization Engine")

# ------------------------------------------------------
# CHECK REQUIRED STATE
# ------------------------------------------------------
required_keys = ["city", "productions", "attractions", "od", "mode_choice"]
missing = [k for k in required_keys if k not in st.session_state]
if missing:
    st.error(f"Please complete earlier steps first. Missing: {', '.join(missing)}")
    st.stop()

city = st.session_state["city"]
taz = city.taz
od_base_dict: Dict[str, pd.DataFrame] = st.session_state["od"]
mode_choice_base = st.session_state["mode_choice"]
tt_car_base = city.travel_time_matrix

# ------------------------------------------------------
# NETWORK
# ------------------------------------------------------
if "network" not in st.session_state:
    network = generate_synthetic_network(taz)
    st.session_state["network"] = network
else:
    network = st.session_state["network"]

od_total_car = mode_choice_base.volumes["car"]


# ------------------------------------------------------
# REUSE POLICY MODE CHOICE LOGIC (copied from Page 8)
# ------------------------------------------------------
def build_policy_time_cost_matrices(
        tt_car_base: pd.DataFrame,
        metro_time_reduction_pct: float,
        metro_fare_change_pct: float,
        congestion_charge: float,
        cbd_zones: list
):
    tt_car = tt_car_base.copy()
    tt_metro = tt_car * 0.8 * (1 - metro_time_reduction_pct / 100.0)
    tt_bus = tt_car * 1.3

    dist_proxy = tt_car / 60 * 30
    cost_car = 2 + 0.12 * dist_proxy
    cost_metro = 15 * (1 + metro_fare_change_pct / 100.0)
    cost_bus = 8 + 0.03 * dist_proxy

    # Vectorized congestion charge
    cost_car.loc[:, cbd_zones] += congestion_charge

    return (
        {"car": tt_car, "metro": tt_metro, "bus": tt_bus},
        {"car": cost_car, "metro": cost_metro, "bus": cost_bus},
    )


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
    total_od = sum(od_mats.values()).loc[zones, zones]

    time_mats, cost_mats = build_policy_time_cost_matrices(
        tt_car_base,
        metro_time_reduction_pct,
        metro_fare_change_pct,
        congestion_charge,
        cbd_zones,
    )

    car_own = taz["car_ownership_rate"].reindex(zones).to_numpy()
    car_own_mat = np.repeat(car_own[:, None], len(zones), axis=1)

    modes = ["car", "metro", "bus"]
    utilities = {}

    for mode in modes:
        tt = time_mats[mode].to_numpy()
        cc = cost_mats[mode].to_numpy()

        if mode == "car":
            U = beta_time * tt + beta_cost * cc + beta_car_own * car_own_mat
        else:
            U = beta_time * tt + beta_cost * cc

        utilities[mode] = U

    exp_sum = sum(np.exp(U) for U in utilities.values())
    probabilities = {
        m: pd.DataFrame(np.exp(U) / np.maximum(exp_sum, 1e-12), index=zones, columns=zones)
        for m, U in utilities.items()
    }

    volumes = {
        m: pd.DataFrame(
            total_od.to_numpy() * probabilities[m].to_numpy(),
            index=zones, columns=zones
        )
        for m in modes
    }

    return probabilities, volumes, total_od


# ------------------------------------------------------
# UI OPTIONS
# ------------------------------------------------------
use_emulator = st.checkbox(
    "Use AI Link Flow Emulator (if trained)",
    value=False
)

st.sidebar.header("Search Space")

mt_min = st.sidebar.slider("Metro time reduction min (%)", 0, 50, 0)
mt_max = st.sidebar.slider("Metro time reduction max (%)", 0, 50, 30)
mt_step = st.sidebar.slider("Metro step (%)", 5, 20, 10)

fare_min = st.sidebar.slider("Metro fare change min (%)", -50, 50, -30)
fare_max = st.sidebar.slider("Metro fare change max (%)", -50, 50, 10)
fare_step = st.sidebar.slider("Metro fare step (%)", 10, 30, 20)

cc_min = st.sidebar.slider("Congestion charge min", 0, 100, 0)
cc_max = st.sidebar.slider("Congestion charge max", 0, 100, 50)
cc_step = st.sidebar.slider("Charge step", 10, 50, 20)

default_cbd = list(taz.index[:5])
cbd_zones = st.sidebar.multiselect(
    "CBD zones",
    options=list(taz.index),
    default=default_cbd,
)

objective_choice = st.selectbox(
    "Optimization Objective",
    ["Minimize total car trips", "Minimize total car link flow"]
)


# ------------------------------------------------------
# OPTIMIZATION ENGINE
# ------------------------------------------------------
if st.button("Run Optimization Search"):
    st.info("Running optimization… may take some time.")

    metro_range = np.arange(mt_min, mt_max + 1e-6, mt_step)
    fare_range = np.arange(fare_min, fare_max + 1e-6, fare_step)
    cc_range = np.arange(cc_min, cc_max + 1e-6, cc_step)

    emulator = st.session_state.get("link_flow_emulator", None)
    results = []

    for mt_red in metro_range:
        for fare_ch in fare_range:
            for cc in cc_range:

                probs, vols, total_od = policy_mode_choice(
                    od_base_dict, taz, tt_car_base,
                    mt_red, fare_ch, cc, cbd_zones
                )

                car_od = vols["car"]
                total_car_trips = car_od.values.sum()

                if use_emulator and emulator is not None:
                    base = od_total_car.values.sum()
                    demand_scale = float(total_car_trips / max(base, 1e-9))
                    df_flows = predict_link_flows(emulator, demand_scale, network)

                    col = "flow_vehph_emulated" if "flow_vehph_emulated" in df_flows.columns else df_flows.columns[-1]
                    total_car_flow = df_flows[col].sum()
                else:
                    df_flows = frank_wolfe_ue(car_od, network, max_iter=30)
                    col = "flow_vehph" if "flow_vehph" in df_flows.columns else df_flows.columns[-1]
                    total_car_flow = df_flows[col].sum()

                objective_value = total_car_trips if objective_choice.startswith("Minimize total car trips") else total_car_flow

                results.append({
                    "metro_time_reduction_pct": mt_red,
                    "metro_fare_change_pct": fare_ch,
                    "congestion_charge": cc,
                    "total_car_trips": total_car_trips,
                    "total_car_flow": total_car_flow,
                    "objective": objective_value
                })

    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values("objective", ascending=True).reset_index(drop=True)

    st.subheader("Top 10 Best Scenarios")
    st.dataframe(res_sorted.head(10))

    # Save results
    os.makedirs("data", exist_ok=True)
    res_sorted.to_csv("data/optimization_results.csv", index=False)

    st.session_state["opt_results"] = res_sorted

    best = res_sorted.iloc[0]
    st.success(
        f"Best scenario: Metro time ↓{best['metro_time_reduction_pct']}%, "
        f"Metro fare {best['metro_fare_change_pct']}%, "
        f"Charge={best['congestion_charge']} → Objective={best['objective']:.2f}"
    )
