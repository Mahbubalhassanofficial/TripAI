import streamlit as st
from modules.mode_choice import mode_choice
import os

st.title("🚈 Mode Choice – Multinomial Logit")

# -----------------------------------------
# CHECK PREVIOUS STEPS
# -----------------------------------------
if "od" not in st.session_state:
    st.error("Please run Trip Distribution first.")
    st.stop()

# -----------------------------------------
# RUN MODE CHOICE
# -----------------------------------------
if st.button("Run Mode Choice"):
    result = mode_choice(
        st.session_state["od"],
        st.session_state["city"].taz,
        st.session_state["city"].travel_time_matrix
    )
    st.session_state["mode_choice"] = result
    st.success("Mode choice completed!")

# -----------------------------------------
# DISPLAY RESULTS
# -----------------------------------------
if "mode_choice" in st.session_state:

    result = st.session_state["mode_choice"]

    st.subheader("Total OD Matrix (all purposes)")
    st.dataframe(result.total_od)

    # ensure save folder exists
    os.makedirs("data", exist_ok=True)

    # save & display mode-specific OD volumes
    for m in result.volumes:
        st.subheader(f"Mode: {m}")
        st.dataframe(result.volumes[m])

        # Save to CSV
        result.volumes[m].to_csv(f"data/od_mode_{m}.csv")

    st.info("Mode-choice outputs saved to /data/")
