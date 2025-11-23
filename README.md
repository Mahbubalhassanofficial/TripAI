
# 📘 **README.md — TripAI: Intelligent Four-Step Travel Demand Modelling**

### **Mahbub Hassan**

**Department of Civil Engineering**
**Faculty of Engineering**
**Chulalongkorn University, Bangkok, Thailand**
Founder, **B’Deshi Emerging Research Lab**

---

# 🚦 **TripAI**

### *A Research-Grade AI-Enhanced Four-Step Travel Demand Modelling Platform*

**TripAI** is a complete, intelligent, open-source platform built for transportation researchers, traffic engineers, and planning professionals.
It integrates classical travel demand modelling with modern AI/XAI techniques and provides a synthetic, reproducible testbed for academic publishing.



---

# 🌐 **Key Features**

### 🏙️ **1. Synthetic City Generator (20-TAZ)**

* Generates a complete synthetic metropolitan region
* Population, households, jobs, income, car-ownership, land-use characteristics
* Distance & travel-time matrices
* Fully reproducible and scientifically grounded

### 🚶 **2. Trip Generation**

* Productions & Attractions for HBW, HBE, HBS
* Balanced with iterative proportional fitting (IPF)

### 🌍 **3. Trip Distribution**

* Doubly-constrained gravity model
* Calibrated friction factor
* Balanced OD matrices

### 🚈 **4. Mode Choice (MNL)**

* Multinomial Logit: Car, Metro, Bus
* Generalized cost structure
* Probability + flow matrices

### 🛣️ **5. Route Assignment**

* All-or-Nothing (AON)
* User Equilibrium (UE) using Frank–Wolfe
* Synthetic network generator

### 🤖 **6. AI-Enhanced Travel Modelling**

* Random Forest Regression for Trip Generation
* Random Forest Classification for Mode Choice
* **SHAP Explainable AI** for behavioural insights
* Publication-ready plots

### 🧠 **7. AI Link Flow Emulator**

* ML-based link flow prediction
* Predict flows under demand scaling without running UE
* Enables ultra-fast scenario analysis

### ⚙️ **8. Policy Scenario Engine**

* Metro travel-time improvements
* Fare reduction/increase
* Congestion charge by destination zone
* TOD-driven attraction uplift
* Baseline vs Scenario comparisons

### 🎯 **9. Scenario Optimization**

* Grid search over policy parameters
* Objective: minimize car trips or network congestion
* Supports emulator or full UE assignment

### 📈 **10. Visualization Dashboard**

* Mode shares
* OD heatmaps
* Car link flow change
* TAZ maps
* 600+ DPI publication-ready figure export

### 📦 **11. Export Module**

* Download all intermediate & final results
* ZIP export for reproducible research pipelines

---

# 📂 **Project Structure**

```
TripAI/
│
├── app.py
├── requirements.txt
│
├── modules/
│   ├── synthetic_city.py
│   ├── trip_generation.py
│   ├── gravity_model.py
│   ├── mode_choice.py
│   ├── route_assignment.py
│   ├── ai_link_flow_emulator.py
│   └── utils.py
│
└── pages/
    ├── 1_📊_Generate_Synthetic_City.py
    ├── 2_🚶_Trip_Generation.py
    ├── 3_🌍_Trip_Distribution.py
    ├── 4_🚈_Mode_Choice.py
    ├── 5_🛣️_Route_Assignment.py
    ├── 6_🤖_AI_Models.py
    ├── 7_⚙️_Policy_Scenario_Engine.py
    ├── 8_📈_Visualization_Dashboard.py
    ├── 9_📦_Export_Results.py
    ├── 10_🧠_AI_Link_Flow_Emulator.py
    └── 11_🎯_Scenario_Optimization.py
```

---

# 🚀 **How to Run (Local)**

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/TripAI.git
cd TripAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

---

# 🆓 **How to Deploy on Free Streamlit Cloud**

1. Push the entire repository to GitHub
2. Go to: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **“Deploy an App”**
4. Select your repository
5. Set:

   * **Main file:** `app.py`
   * **Python version:** Auto
   * **Requirements file:** `requirements.txt`

Streamlit Cloud will install everything and deploy automatically.

---

# 📜 **Citation (APA 7th)**

If you use TripAI in a publication:

> Hassan, M. (2025). *TripAI: Intelligent Four-Step Travel Demand Modelling with AI, XAI, and Scenario Optimization*. B’Deshi Emerging Research Lab, Department of Civil Engineering, Chulalongkorn University.

---

# 💡 **Future Extensions**

* Large-scale networks
* GTFS-based multimodal assignment
* LLM-driven behavioural modelling
* Federated learning for distributed travel survey data
* Optimization using reinforcement learning

---

# 🤝 **Contact**

**Mahbub Hassan**
Graduate Research Student
Department of Civil Engineering
Faculty of Engineering
**Chulalongkorn University, Thailand**
email: mahbub.hassan@ieee.org; 6870376421@student.chula.ac.th
Founder, **B’Deshi Emerging Research Lab**

---

