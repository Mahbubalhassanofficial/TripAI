import streamlit as st
import pandas as pd
import os

from modules.synthetic_city import generate_synthetic_city

st.title("📊 Generate Synthetic City (20 TAZ)")

# -------------------------------------
# GENERATE SYNTHETIC CITY
# -------------------------------------
if st.button("Generate Synthetic Region"):
    city = generate_synthetic_city()

    # Save to session state
    st.session_state["city"] = city
    st.success("Synthetic city generated successfully!")

# -------------------------------------
# DISPLAY RESULTS
# -------------------------------------
if "city" in st.session_state:
    city = st.session_state["city"]

    st.subheader("TAZ Attributes")
    st.dataframe(city.taz)

    st.subheader("Summary Statistics")
    st.write(city.taz.describe())

    # Save files to /data/
    os.makedirs("data", exist_ok=True)
    city.taz.to_csv("data/taz_attributes.csv")
    city.distance_matrix.to_csv("data/distance_matrix.csv")
    city.travel_time_matrix.to_csv("data/travel_time_matrix.csv")

    st.info("Files saved in /data/")
