import streamlit as st
from modules.trip_generation import trip_generation
import pandas as pd
import os

st.title("🚶 Trip Generation")

if "city" not in st.session_state:
    st.error("Please generate the synthetic city first.")
    st.stop()

city = st.session_state["city"]

if st.button("Run Trip Generation"):
    P, A = trip_generation(city.taz)
    st.session_state["productions"] = P
    st.session_state["attractions"] = A
    st.success("Trip generation completed!")

if "productions" in st.session_state:
    P = st.session_state["productions"]
    A = st.session_state["attractions"]

    st.subheader("Productions")
    st.dataframe(P)

    st.subheader("Attractions (Balanced)")
    st.dataframe(A)

    os.makedirs("data", exist_ok=True)
    P.to_csv("data/productions.csv")
    A.to_csv("data/attractions.csv")
    st.info("Saved to /data/")
