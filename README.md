# 🚦 TripAI  
### Intelligent Four-Step Travel Demand Modelling with AI, XAI, and Optimization  
Developed by Mahbub Hassan, B’Deshi Emerging Research Lab

TripAI is a **Streamlit-powered research system** that implements a full  
**synthetic four-step travel demand model**, enhanced with modern  
**Machine Learning**, **Explainable AI (SHAP)**,  
**User Equilibrium (Frank–Wolfe)**, and  
**Policy Optimization Tools**.

TripAI is suitable for:
- Transportation engineering research  
- AI-driven mobility modeling  
- Master’s/PhD coursework  
- Q1 journal publications  
- Decision-support & teaching  

---

## 🔧 Core Features

### 🏙 1. Synthetic City Generator
- Auto-creates a 20-TAZ synthetic city  
- Population, workers, students, income, car ownership  
- Distances & travel time matrices  

### 🚶 2. Trip Generation  
- HBW, HBE, HBS production/attraction models  
- Full balancing and control totals  

### 🌍 3. Trip Distribution  
- Gravity model  
- IPF balancing  
- Purpose-specific impedance  

### 🚈 4. Mode Choice  
- Multinomial logit (Car, Metro, Bus)  
- Cost, time, car ownership effects  

### 🛣 5. Route Assignment  
- All-or-Nothing  
- **User Equilibrium (Frank–Wolfe)**  

### 🤖 6. AI & XAI  
- ML Trip Generation (RandomForestRegressor)  
- ML Mode Choice (RandomForestClassifier)  
- SHAP global interpretability  

### 🧠 7. AI Link Flow Emulator  
- Multi-output RF surrogate for link flows  
- Predict flows instantly without running UE  

### ⚙ 8. Policy Scenario Engine  
- Metro improvements  
- Fare changes  
- Congestion pricing  
- TOD-based attraction changes  

### 🎯 9. Scenario Optimization  
- Search for best policy combination  
- Objective: minimize car use or congestion  

### 📈 10. Visualization Dashboard  
- OD heatmaps  
- Mode share comparison  
- Car flow change charts  
- TAZ spatial maps  
- 600-DPI export  

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
