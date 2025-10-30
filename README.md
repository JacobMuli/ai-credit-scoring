# 🌾 AI Credit Scoring for Smallholder Farmers

### 🧩 Developed for the **AI 4 Startups Hackathon (Data Governance in Africa)**  
**Tracks Covered:**  
✅ Track 1 — AI Model Design  
✅ Track 2 — Risk & Loan Simulation Engine  

---

## 🎯 Project Overview

Smallholder farmers often lack formal financial histories, making it difficult for them to access loans.  
This project builds an **AI-powered credit scoring system** that leverages **alternative and synthetic data** to predict farmers’ creditworthiness and simulate real-world lending decisions.

The system is designed to:
- Support **inclusive agricultural financing**
- Enhance **risk transparency**
- Promote **ethical and explainable AI adoption** in Agri-FinTech

---

## 🚀 Features

### 🔹 Track 1: AI Model
- Machine Learning model trained on **synthetic farmer data** generated using `Faker` and `NumPy`.
- Predicts **default probability** and **credit score**.
- Features include:
  - Demographics (Gender, Age)
  - Agricultural data (Crop Type, Farm Size, Yield History)
  - Financial data (Mobile Money Transactions, Wallet Balance)
  - Environmental data (NDVI, Drought Exposure)
  - Behavioral indicators (Cooperative Membership)

**Evaluation Metrics:**  
ROC-AUC, F1-Score, Precision, Recall, and Confusion Matrix

**Model Architecture:**  
Random Forest Classifier with feature preprocessing pipeline.

---

### 🔹 Track 2: Simulation Engine (Streamlit App)

The app (`app.py`) provides a fully interactive dashboard with:

#### 🧾 Credit Score Prediction
- Compute **credit score**, **default probability**, and **loan eligibility**.  
- Dynamic computation of **interest rates** and **suggested loan amounts**.

#### ⚙️ Risk & What-If Simulation
- Simulate real-world shocks like:
  - Drought exposure  
  - NDVI degradation  
  - Market price drops  
  - Delayed repayments  
- Observe instant changes in **credit score** and **eligibility**.

#### 💰 Repayment Schedule Simulator
- Generates a **monthly amortization table** based on loan amount, interest rate, and duration.  
- Visualizes **repayment progress** and **balance reduction** over time.

#### 📊 Model Performance Dashboard
- Displays ROC Curve, Confusion Matrix, and Top 10 Feature Importances.  
- Evaluates explainability and fairness of model predictions.

---

## 🧠 Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python |
| **Frameworks** | Streamlit, Scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | Feature Importance (SHAP-ready) |
| **Hosting** | Streamlit Cloud / GitHub Pages |

---

## ⚙️ Setup & Run Locally

```bash
# 1️⃣ Clone repository
git clone https://github.com/JacobMuli/ai-credit-scoring.git
cd ai-credit-scoring

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run Streamlit app
streamlit run app.py

📁 ai-credit-scoring/
├── app.py                         # Streamlit simulation app
├── credit_model.pkl.gz            # Trained AI model (compressed)
├── main_harmonized_dataset_final.csv  # Harmonized dataset
├── requirements.txt               # Dependencies
├── README.md                      # Project overview
└── colab_notebook_py.ipynb        # Model training notebook

🌍 Impact

This solution supports financial inclusion by offering:

AI-driven, transparent, and scalable credit scoring.

Risk-based loan simulation for more adaptive agricultural lending.

A reproducible and open-source pipeline that aligns with Responsible AI in Agri-Finance.
