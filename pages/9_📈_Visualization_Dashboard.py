import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(layout="wide")
st.title("📈 Visualization Dashboard")

st.markdown("""
This dashboard provides **research-grade visualizations** for:
- Mode share comparison (Baseline vs Scenario)
- OD heatmaps
- Car flow changes on network links
- TAZ-level spatial indicators  

All figures are exportable in 600 DPI for Q1-grade publications.
""")

# ------------------------------------------------------
# CHECK REQUIRED STATE
# ------------------------------------------------------
if "city" not in st.session_state:
    st.error("Generate synthetic city first.")
    st.stop()

if "mode_choice" not in st.session_state:
    st.error("Complete Mode Choice first.")
    st.stop()

city = st.session_state["city"]
taz = city.taz

# Baseline
mode_base = st.session_state["mode_choice"]
car_flow_base = st.session_state.get("link_flows", None)

# Scenario check
scenario_exists = (
    "vols_scen" in st.session_state and
    "total_od_scen" in st.session_state and
    "link_flows_scenario" in st.session_state
)

# ------------------------------------------------------
# Helper: Export figure in 600 DPI
# ------------------------------------------------------
def export_fig(fig, filename):
    fig.savefig(filename, dpi=600, bbox_inches="tight")
    with open(filename, "rb") as f:
        st.download_button(
            label="⬇ Download Figure",
            data=f,
            file_name=filename,
            mime="image/png"
        )
    st.success("Figure exported in 600 DPI!")

# ======================================================
# SECTION 1: MODE SHARE COMPARISON
# ======================================================
st.header("🚈 Mode Share Comparison (Baseline vs Scenario)")

def compute_mode_shares(mode_volumes, total_od):
    total_trips = total_od.values.sum()
    rows = []
    for m, mat in mode_volumes.items():
        trips = mat.values.sum()
        share = trips / total_trips if total_trips > 0 else 0
        rows.append([m, trips, share])
    return pd.DataFrame(rows, columns=["Mode", "Trips", "Share"])

baseline_shares = compute_mode_shares(
    mode_base.volumes, mode_base.total_od
)

if scenario_exists:
    vols_scen = st.session_state["vols_scen"]
    total_od_scen = st.session_state["total_od_scen"]
    scenario_shares = compute_mode_shares(vols_scen, total_od_scen)

colA, colB = st.columns(2)

with colA:
    st.subheader("Baseline Mode Shares")
    st.dataframe(baseline_shares.style.format({"Trips": "{:,.1f}", "Share": "{:.3f}"}))

with colB:
    if scenario_exists:
        st.subheader("Scenario Mode Shares")
        st.dataframe(scenario_shares.style.format({"Trips": "{:,.1f}", "Share": "{:.3f}"}))
    else:
        st.info("Run a policy scenario to enable comparison.")

# Bar chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(baseline_shares["Mode"], baseline_shares["Share"], label="Baseline")

if scenario_exists:
    ax.bar(
        np.arange(len(scenario_shares)) + 0.3,
        scenario_shares["Share"],
        width=0.3,
        label="Scenario"
    )

ax.set_ylabel("Mode Share")
ax.set_title("Baseline vs Scenario Mode Shares")
ax.legend()
st.pyplot(fig)

export_fig(fig, "mode_share_comparison.png")

# ======================================================
# SECTION 2: OD HEATMAPS
# ======================================================
st.header("🌍 OD Heatmaps (Baseline & Scenario)")

od_base = st.session_state["od"]
purpose = st.selectbox("Select Trip Purpose", list(od_base.keys()))

# Baseline heatmap
st.subheader(f"Baseline OD – {purpose}")

fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.heatmap(od_base[purpose], cmap="viridis", ax=ax2)
ax2.set_title(f"Baseline OD – {purpose}")
st.pyplot(fig2)
export_fig(fig2, f"baseline_od_{purpose}.png")

# Scenario heatmap
if scenario_exists:
    od_scenario = st.session_state.get("od_scenario", None)

    if isinstance(od_scenario, dict) and purpose in od_scenario:
        od_scen_matrix = od_scenario[purpose]
    else:
        od_scen_matrix = od_base[purpose]

    st.subheader(f"Scenario OD – {purpose}")

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.heatmap(od_scen_matrix, cmap="viridis", ax=ax3)
    ax3.set_title(f"Scenario OD – {purpose}")
    st.pyplot(fig3)
    export_fig(fig3, f"scenario_od_{purpose}.png")

# ======================================================
# SECTION 3: CAR FLOW COMPARISON
# ======================================================
st.header("🚗 Car Link Flows (Baseline vs Scenario)")

if car_flow_base is None:
    st.info("Baseline link flows unavailable. Run Route Assignment first.")
else:
    st.subheader("Baseline Link Flows")
    st.dataframe(car_flow_base.head(10))

    if scenario_exists:
        car_flow_scen = st.session_state["link_flows_scenario"]

        st.subheader("Scenario Link Flows")
        st.dataframe(car_flow_scen.head(10))

        # Safe merged comparison
        merged = car_flow_base.copy()
        merged["scenario"] = car_flow_scen.iloc[:, -1]
        merged["change"] = merged["scenario"] - merged.iloc[:, -1]

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.bar(
            merged.index,
            merged["change"],
            color=["red" if x > 0 else "green" for x in merged["change"]]
        )
        ax4.set_title("Change in Car Link Flows")
        ax4.set_ylabel("Δ Flow (veh/h)")
        st.pyplot(fig4)

        export_fig(fig4, "car_link_flow_change.png")

# ======================================================
# SECTION 4: TAZ SPATIAL MAPS
# ======================================================
st.header("🗺️ TAZ-Level Spatial Indicators")

indicator = st.selectbox(
    "Select variable to map",
    ["population", "workers", "students", "land_use_mix", "cars"]
)

fig5, ax5 = plt.subplots(figsize=(6, 6))
scatter = ax5.scatter(
    taz["x_km"], taz["y_km"],
    c=taz[indicator],
    s=220,
    cmap="plasma",
    edgecolors="black"
)
plt.colorbar(scatter, ax=ax5, label=indicator)
ax5.set_title(f"TAZ Map – {indicator.capitalize()}")
st.pyplot(fig5)

export_fig(fig5, f"taz_map_{indicator}.png")

st.success("Visualization dashboard ready.")
