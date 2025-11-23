# pages/10_🧠_AI_Link_Flow_Emulator.py

import streamlit as st
import pandas as pd
import numpy as np
import os

from modules.ai_link_flow_emulator import (
    train_link_flow_emulator,
    predict_link_flows,
)
from modules.route_assignment import generate_synthetic_network

st.set_page_config(layout="wide")
st.title("🧠 AI Link Flow Emulator")

# -----------------------------------------------------
# CHECK REQUIRED STATE
# -----------------------------------------------------
if "mode_choice" not in st.session_state:
    st.error("Run Mode Choice first (Page 4).")
    st.stop()

if "city" not in st.session_state:
    st.error("Generate synthetic city first (Page 1).")
    st.stop()

city = st.session_state["city"]
taz = city.taz
mode_choice = st.session_state["mode_choice"]

# Use car OD as baseline demand
base_car_od = mode_choice.volumes["car"]
base_car_od_np = base_car_od.to_numpy()

# -----------------------------------------------------
# NETWORK
# -----------------------------------------------------
if "network" not in st.session_state:
    network = generate_synthetic_network(taz)
    st.session_state["network"] = network
else:
    network = st.session_state["network"]

st.markdown("### Network Summary")
st.write(f"Number of links: **{len(network)}**")

# -----------------------------------------------------
# TRAINING SCENARIOS
# -----------------------------------------------------
n_scenarios = st.slider(
    "Number of training scenarios to generate",
    min_value=5, max_value=100, value=20, step=1,
    help="More scenarios → better emulator accuracy, slower training."
)

if st.button("Train Emulator"):
    st.info("Training emulator… please wait.")

    emulator, training_history = train_link_flow_emulator(
        base_car_od_np,
        network,
        n_scenarios=n_scenarios
    )

    st.session_state["link_flow_emulator"] = emulator
    st.session_state["emulator_training_history"] = training_history

    st.success("AI Link Flow Emulator trained successfully!")

# -----------------------------------------------------
# PREDICTION MODULE
# -----------------------------------------------------
if "link_flow_emulator" in st.session_state:
    emulator = st.session_state["link_flow_emulator"]

    st.header("📡 Predict New Link Flows with AI")
    scale = st.slider(
        "Demand scaling factor",
        min_value=0.5, max_value=1.5, value=1.0, step=0.05,
        help="Scale baseline OD (e.g., 1.2 = +20% demand)"
    )

    if st.button("Predict Link Flows"):
        pred_df = predict_link_flows(emulator, scale, network)

        st.subheader("AI Predicted Link Flows (sample)")
        st.dataframe(pred_df.head(12))

        # Save output
        os.makedirs("data", exist_ok=True)
        pred_df.to_csv("data/emulator_predicted_link_flows.csv", index=False)
        st.success("Predicted flows saved to /data/emulator_predicted_link_flows.csv")

        # Download button
        csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Predicted Link Flows (CSV)",
            data=csv_bytes,
            file_name="predicted_link_flows_ai.csv",
            mime="text/csv"
        )

else:
    st.info("Train the emulator to enable AI predictions.")
