import streamlit as st
import pandas as pd
import os
import zipfile
from io import BytesIO

st.set_page_config(layout="wide")
st.title("📦 Export Results")

st.markdown("""
This module allows you to **download full outputs** from all steps of the Four-Step Model:

- Synthetic City (TAZ data)
- Trip Generation
- Trip Distribution (OD matrices)
- Mode Choice (volumes + probabilities)
- Route Assignment (link flows)
- AI Model Outputs (regression, classification)
""")

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# ------------------------------------------------------
# Helper: Save CSV to buffer
# ------------------------------------------------------
def make_csv_download(df: pd.DataFrame, filename: str):
    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label=f"⬇ Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# ------------------------------------------------------
# Helper: Create ZIP file dynamically
# ------------------------------------------------------
def build_zip_file():
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:

        if "city" in st.session_state:
            city = st.session_state["city"]
            z.writestr("taz_attributes.csv", city.taz.to_csv())
            z.writestr("distance_matrix.csv", city.distance_matrix.to_csv())
            z.writestr("travel_time_matrix.csv", city.travel_time_matrix.to_csv())

        if "productions" in st.session_state:
            z.writestr("productions.csv", st.session_state["productions"].to_csv())
        if "attractions" in st.session_state:
            z.writestr("attractions.csv", st.session_state["attractions"].to_csv())

        if "od" in st.session_state:
            for p, df in st.session_state["od"].items():
                z.writestr(f"od_{p}.csv", df.to_csv())

        if "mode_choice" in st.session_state:
            mc = st.session_state["mode_choice"]
            z.writestr("total_od.csv", mc.total_od.to_csv())
            for m, df in mc.volumes.items():
                z.writestr(f"od_mode_{m}.csv", df.to_csv())
            for m, df in mc.probabilities.items():
                z.writestr(f"mode_prob_{m}.csv", df.to_csv())

        if "link_flows" in st.session_state:
            link_flows = st.session_state["link_flows"]
            z.writestr("link_flows.csv", link_flows.to_csv())

        # AI models not serializable → export predictions & metadata only
        if "ai_tripgen_model" in st.session_state:
            model = st.session_state["ai_tripgen_model"]
            city = st.session_state["city"]
            preds = model.predict(city.taz[[
                "population","households","workers","students",
                "income","car_ownership_rate","land_use_mix",
                "service_jobs","industrial_jobs","retail_jobs"
            ]])
            pred_df = pd.DataFrame(preds, index=city.taz.index,
                                   columns=["AI_TripGen_Pred"])
            z.writestr("ai_trip_generation_predictions.csv", pred_df.to_csv())

        if "ai_modechoice_model" in st.session_state:
            clf = st.session_state["ai_modechoice_model"]
            z.writestr("ai_modechoice_classes.txt",
                       "\n".join(list(clf.classes_)))

    buffer.seek(0)
    return buffer

# ------------------------------------------------------
# DISPLAY DOWNLOAD SECTION
# ------------------------------------------------------

st.header("📁 Download Individual Outputs")

# Synthetic City
if "city" in st.session_state:
    st.subheader("🏙️ Synthetic City")
    make_csv_download(st.session_state["city"].taz, "taz_attributes.csv")
    make_csv_download(st.session_state["city"].distance_matrix, "distance_matrix.csv")
    make_csv_download(st.session_state["city"].travel_time_matrix, "travel_time_matrix.csv")
else:
    st.info("Synthetic city not generated yet.")

# Trip Generation
if "productions" in st.session_state:
    st.subheader("🚶 Trip Generation")
    make_csv_download(st.session_state["productions"], "productions.csv")
    make_csv_download(st.session_state["attractions"], "attractions.csv")

# Trip Distribution
if "od" in st.session_state:
    st.subheader("🌍 Trip Distribution – OD Matrices")
    for purpose, df in st.session_state["od"].items():
        make_csv_download(df, f"od_{purpose}.csv")

# Mode Choice
if "mode_choice" in st.session_state:
    st.subheader("🚈 Mode Choice – Volumes & Probabilities")
    mc = st.session_state["mode_choice"]
    make_csv_download(mc.total_od, "total_od.csv")
    for m, df in mc.volumes.items():
        make_csv_download(df, f"od_mode_{m}.csv")
    for m, df in mc.probabilities.items():
        make_csv_download(df, f"mode_prob_{m}.csv")

# Route Assignment
if "link_flows" in st.session_state:
    st.subheader("🛣️ Route Assignment – Link Flows")
    make_csv_download(st.session_state["link_flows"], "link_flows.csv")

# AI Models
if "ai_tripgen_model" in st.session_state or "ai_modechoice_model" in st.session_state:
    st.subheader("🤖 AI Model Outputs")

    if "ai_tripgen_model" in st.session_state:
        st.write("• AI Trip Generation model predictions available")

    if "ai_modechoice_model" in st.session_state:
        st.write("• AI Mode Choice classifier classes available")

# ------------------------------------------------------
# ZIP EXPORT
# ------------------------------------------------------

st.header("📦 Download EVERYTHING (ZIP)")

if st.button("Create ZIP Package"):
    zip_buffer = build_zip_file()
    st.download_button(
        label="⬇ Download Zip File",
        data=zip_buffer,
        file_name="TripAI_outputs.zip",
        mime="application/zip"
    )
    st.success("ZIP file prepared!")
