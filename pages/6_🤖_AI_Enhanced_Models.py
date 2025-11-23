import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.set_page_config(layout="wide")
st.title("🤖 AI-Enhanced Four-Step Model")

st.markdown("""
This module introduces **Machine Learning + Explainable AI (XAI)** to improve:
- **Trip Generation (Regression Model)**
- **Mode Choice (Classification Model)**
- **Behavioral Interpretation using SHAP**

Use this page *after* completing Steps 1–5.
""")

# -------------------------------------------------------
# CHECK DATA
# -------------------------------------------------------
if "city" not in st.session_state:
    st.error("Please generate the synthetic city first (Page 1).")
    st.stop()

if "productions" not in st.session_state:
    st.error("Please complete Trip Generation (Page 2).")
    st.stop()

if "mode_choice" not in st.session_state:
    st.error("Please complete Mode Choice (Page 4).")
    st.stop()

# Load needed data
taz = st.session_state["city"].taz
productions = st.session_state["productions"]
mode_choice_result = st.session_state["mode_choice"]

# -------------------------------------------------------
# SECTION 1 — AI Trip Generation (Regression)
# -------------------------------------------------------
st.header("🚶 AI-based Trip Generation (Regression)")

purpose = st.selectbox(
    "Select Trip Purpose to Model",
    ["HBW", "HBE", "HBS"]
)

X = taz[[
    "population", "households", "workers", "students",
    "income", "car_ownership_rate", "land_use_mix",
    "service_jobs", "industrial_jobs", "retail_jobs"
]]

y = productions[purpose]

if st.button("Train AI Trip Generation Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.success("AI Regression Model Trained!")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**R²:** {r2:.3f}")

    st.session_state["ai_tripgen_model"] = model

    st.subheader("🔍 SHAP Explanation of Trip Generation Model")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    shap.plots.bar(shap_values, max_display=10, show=False)
    fig = plt.gcf()
    st.pyplot(fig)

# -------------------------------------------------------
# SECTION 2 — AI Mode Choice (Classification)
# -------------------------------------------------------
st.header("🚈 AI-based Mode Choice (Classification)")

vol = mode_choice_result.volumes
P = mode_choice_result.probabilities

rows = []
zones = list(taz.index)
TT = st.session_state["city"].travel_time_matrix

for i in zones:
    for j in zones:
        if i == j:
            continue

        probs = [
            P["car"].loc[i, j],
            P["metro"].loc[i, j],
            P["bus"].loc[i, j]
        ]

        # Normalize probabilities (avoid zero-sum)
        s = sum(probs)
        if s == 0:
            continue
        probs = [p / s for p in probs]

        label = np.random.choice(["car", "metro", "bus"], p=probs)

        rows.append({
            "origin": i,
            "destination": j,
            "travel_time": TT.loc[i, j],
            "car_ownership": float(taz.loc[i, "car_ownership_rate"]),
            "cost_car": 2 + 0.1 * TT.loc[i, j],
            "cost_metro": 15,
            "cost_bus": 8,
            "label": label
        })

df_mc = pd.DataFrame(rows)

feature_cols = ["travel_time", "car_ownership", "cost_car", "cost_metro", "cost_bus"]
X_mc = df_mc[feature_cols]
y_mc = df_mc["label"]

if st.button("Train AI Mode Choice Classifier"):
    X_train, X_test, y_train, y_test = train_test_split(
        X_mc, y_mc, test_size=0.25, random_state=42, stratify=y_mc
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success("AI Mode Choice Classifier Trained!")
    st.write(f"**Accuracy:** {acc:.3f}")

    st.session_state["ai_modechoice_model"] = clf

    st.subheader("🔍 SHAP Explanation for Mode Choice")

    explainer = shap.Explainer(clf, X_train)
    shap_values = explainer(X_train)

    shap.plots.bar(shap_values, max_display=10, show=False)
    fig2 = plt.gcf()
    st.pyplot(fig2)

# -------------------------------------------------------
# SECTION 3 — Summary
# -------------------------------------------------------
st.header("📘 Interpretation Summary")

st.markdown("""
### ✔ Completed:
- **AI Regression for Trip Generation**
- **AI Classification for Mode Choice**
- **SHAP-based Explainability**

### ✔ Enables:
- Hybrid classical–AI modelling  
- Behavioral insights  
- Scenario sensitivity  
- Publishable Q1-grade figures  
""")
