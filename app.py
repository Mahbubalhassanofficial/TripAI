# app.py
# TripAI – Intelligent Four-Step Travel Demand Modelling
# Main Entry Point for the Multi-Page Streamlit Application

import streamlit as st

st.set_page_config(
    page_title="TripAI – Intelligent Four-Step Travel Demand Model",
    page_icon="🚦",
    layout="wide"
)

# ==========================================================
# HEADER
# ==========================================================
st.title("🚦 TripAI")
st.markdown("### Intelligent Four-Step Travel Demand Modelling with AI, XAI, and Optimization")

st.markdown(
    """
TripAI is a **research-oriented platform** implementing a complete, synthetic  
**four-step travel demand model**, augmented with:

- Classical **Trip Generation → Trip Distribution → Mode Choice → Route Assignment**
- **User Equilibrium (UE)** using Frank–Wolfe
- **Machine Learning** (Regression + Classification)  
- **Explainable AI** (SHAP) for behavioural insights
- **AI Link Flow Emulator** for fast demand scaling
- **Policy Scenario Engine** with congestion charge, TOD, MRT improvements
- **Scenario Optimization** over policy parameters

Use the **left sidebar** to navigate between phases of the workflow.
"""
)

# ==========================================================
# SESSION STATUS PANEL
# ==========================================================
st.markdown("---")
st.subheader("📊 Current Session Status")

col1, col2, col3 = st.columns(3)

# ----- Column 1 -----
with col1:
    st.markdown("**1. Synthetic City**")
    if "city" in st.session_state:
        taz = st.session_state["city"].taz
        st.success(f"Generated ({len(taz)} TAZs)")
        st.caption("Go to: `📊 Generate Synthetic City`")
    else:
        st.info("Not generated")

    st.markdown("**2. Trip Generation**")
    if "productions" in st.session_state and "attractions" in st.session_state:
        st.success("Done")
        st.caption("Go to: `🚶 Trip Generation`")
    else:
        st.info("Not run")

# ----- Column 2 -----
with col2:
    st.markdown("**3. Trip Distribution**")
    if "od" in st.session_state:
        st.success("OD matrices available")
        st.caption("Go to: `🌍 Trip Distribution`")
    else:
        st.info("Not run")

    st.markdown("**4. Mode Choice**")
    if "mode_choice" in st.session_state:
        st.success("Mode choice available")
        st.caption("Go to: `🚈 Mode Choice`")
    else:
        st.info("Not run")

# ----- Column 3 -----
with col3:
    st.markdown("**5. Route Assignment**")
    if "link_flows" in st.session_state:
        st.success("Assignment complete")
        st.caption("Go to: `🛣️ Route Assignment`")
    else:
        st.info("Not run")

    st.markdown("**6. AI / Scenario / Visualization**")

    status = []
    if "ai_tripgen_model" in st.session_state:
        status.append("AI TripGen")
    if "ai_modechoice_model" in st.session_state:
        status.append("AI ModeChoice")
    if "link_flow_emulator" in st.session_state:
        status.append("AI Emulator")
    if "opt_results" in st.session_state:
        status.append("Optimization")

    if status:
        st.success(" / ".join(status))
        st.caption("See: `🤖 AI`, `🧠 Emulator`, `🎯 Optimization`, `📈 Visualization`")
    else:
        st.info("No AI/Scenario modules executed")

# ==========================================================
# WORKFLOW EXPLANATION
# ==========================================================
st.markdown("---")
st.subheader("🧭 Recommended Workflow")

st.markdown(
    """
1. **📊 Generate Synthetic City**  
   Build a 20-zone synthetic metro with socio-economic + land-use attributes.

2. **🚶 Trip Generation**  
   Compute productions & attractions for HBW, HBE, HBS.

3. **🌍 Trip Distribution**  
   Doubly-constrained gravity model with IPF.

4. **🚈 Mode Choice**  
   Multinomial Logit (Car / Metro / Bus).

5. **🛣️ Route Assignment**  
   AON or User Equilibrium (Frank–Wolfe).

6. **🤖 AI-Enhanced Models**  
   ML Regression + Classification + SHAP explanations.

7. **⚙️ Policy Scenario Engine**  
   Metro improvements, congestion charge, fare changes, TOD.

8. **🧠 AI Link Flow Emulator**  
   Predict link flows without running UE.

9. **🎯 Scenario Optimization**  
   Search policy space to minimize congestion or car use.

10. **📈 Visualization & 📦 Export**  
    Create research-grade figures & download complete datasets.
"""
)

st.markdown("---")
st.caption("TripAI – Developed by Mahbub Hassan, B’Deshi Emerging Research Lab.")
