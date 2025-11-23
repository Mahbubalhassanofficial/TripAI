import streamlit as st
from modules.gravity_model import build_all_od_matrices
import pandas as pd
import os

st.title("🌍 Trip Distribution – Gravity Model")

# --------------------------------------------------
# CHECK PREVIOUS STEPS
# --------------------------------------------------
if "productions" not in st.session_state:
    st.error("Please complete Trip Generation first.")
    st.stop()

# --------------------------------------------------
# RUN GRAVITY MODEL
# --------------------------------------------------
if st.button("Run Gravity Model"):
    P = st.session_state["productions"]
    A = st.session_state["attractions"]
    TT = st.session_state["city"].travel_time_matrix

    od_mats = build_all_od_matrices(P, A, TT)

    st.session_state["od"] = od_mats
    st.success("Trip distribution completed!")

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
if "od" in st.session_state:
    od = st.session_state["od"]

    os.makedirs("data", exist_ok=True)

    for purpose, mat in od.items():
        st.subheader(f"OD Matrix – {purpose}")
        st.dataframe(mat)

        mat.to_csv(f"data/od_{purpose}.csv")

    st.info("OD matrices saved to /data/")
