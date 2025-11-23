# pages/5_🛣️_Route_Assignment.py

import streamlit as st
import os

from modules.route_assignment import (
    generate_synthetic_network,
    aon_assignment,
    frank_wolfe_ue,
)

st.title("🛣️ Route Assignment – AON & UE")

# ------------------------------------------------
# CHECK MODE CHOICE
# ------------------------------------------------
if "mode_choice" not in st.session_state:
    st.error("Run Mode Choice first (Page 4).")
    st.stop()

mode_choice = st.session_state["mode_choice"]
city = st.session_state["city"]
taz = city.taz

# ------------------------------------------------
# CHOOSE METHOD
# ------------------------------------------------
assignment_type = st.selectbox(
    "Select assignment method",
    ["All-or-Nothing (AON)", "User Equilibrium (UE – Frank–Wolfe)"]
)

# ------------------------------------------------
# RUN ASSIGNMENT
# ------------------------------------------------
if st.button("Generate Network & Run Assignment"):

    # Generate or load network
    if "network" in st.session_state:
        network = st.session_state["network"]
    else:
        network = generate_synthetic_network(taz)
        st.session_state["network"] = network

    # Use car OD matrix only
    car_od = mode_choice.volumes["car"]

    if assignment_type.startswith("All"):
        link_flows = aon_assignment(car_od, network)
        st.session_state["link_flows"] = link_flows
        st.success("All-or-Nothing assignment completed.")
    else:
        link_flows_ue = frank_wolfe_ue(car_od, network)
        st.session_state["link_flows"] = link_flows_ue
        st.success("User Equilibrium (Frank–Wolfe) assignment completed.")

# ------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------
if "link_flows" in st.session_state:
    st.subheader("Assigned Link Flows (sample)")
    st.dataframe(st.session_state["link_flows"].head(12))

    # Save to /data/
    os.makedirs("data", exist_ok=True)
    st.session_state["link_flows"].to_csv("data/link_flows.csv")
    st.info("Link flows saved to /data/")
