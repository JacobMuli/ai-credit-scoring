import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns
import pickle, os, numpy as np

st.set_page_config(page_title="🌾 AI Credit Scoring", layout="wide")

# Load model
MODEL_PATH = "credit_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Tabs for navigation
tab1, tab2 = st.tabs(["🧾 Predict Farmer Score", "📊 Dashboard"])

# --- Tab 1: Prediction ---
with tab1:
    st.sidebar.header("📋 Farmer Profile")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 80, 35)
    farm_size = st.sidebar.number_input("Farm size (hectares)", 0.1, 100.0, 3.5, step=0.1)
    crop = st.sidebar.selectbox("Main Crop", ["Maize", "Beans", "Tea", "Coffee", "Horticulture"])
    cooperative = st.sidebar.selectbox("Member of Cooperative", [0, 1])
    yield_hist = st.sidebar.number_input("Average yield (tons/ha)", 0.1, 10.0, 2.5, step=0.1)
    mobile_txns = st.sidebar.number_input("Monthly Mobile Transactions", 0, 200, 25)
    mobile_balance = st.sidebar.number_input("Avg. Mobile Wallet Balance (KES)", 0, 100000, 1500)
    ndvi = st.sidebar.slider("NDVI (Vegetation Health)", 0.05, 0.9, 0.55, step=0.01)
    drought_exposure = st.sidebar.selectbox("Drought Exposure (recent)", [0, 1])

    sample = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "farm_size": farm_size,
        "crop": crop,
        "cooperative": cooperative,
        "yield_hist": yield_hist,
        "mobile_txns": mobile_txns,
        "mobile_balance": mobile_balance,
        "ndvi": ndvi,
        "drought_exposure": drought_exposure
    }])

    if st.button("🚀 Predict Credit Score"):
        prob_default = model.predict_proba(sample)[0, 1]
        credit_score = (1 - prob_default) * 1000
        st.metric("Credit Score", f"{credit_score:.0f}")
        st.metric("Default Probability", f"{prob_default:.2%}")

# --- Tab 2: Dashboard ---
with tab2:
    st.header("📊 Credit Scoring Model Dashboard")

    st.write("Showing ROC Curve and Confusion Matrix using synthetic data...")
    def generate_farmers(n=1000, seed=42):
        rng = np.random.RandomState(seed)
        gender = rng.choice(['Male', 'Female'], size=n, p=[0.6, 0.4])
        age = rng.randint(18, 70, size=n)
        farm_size = np.round(np.exp(rng.normal(np.log(2.0), 0.8, size=n)), 2)
        crop = rng.choice(['Maize', 'Beans', 'Tea', 'Coffee', 'Horticulture'], size=n)
        cooperative = rng.binomial(1, 0.4, n)
        yield_hist = np.maximum(0.1, rng.normal(2.0, 0.8, size=n))
        mobile_txns = rng.poisson(25, size=n)
        mobile_balance = np.maximum(0, rng.normal(1200, 600, size=n))
        ndvi = np.clip(rng.normal(0.45 + 0.1*(yield_hist/3.0), 0.08), 0.05, 0.9)
        drought_exposure = rng.binomial(1, 0.3, size=n)
        logits = (
            -1.0 * np.log(farm_size + 0.1)
            - 0.8 * cooperative
            - 1.2 * (yield_hist / np.maximum(farm_size, 0.1))
            + 1.5 * drought_exposure
            - 0.001 * mobile_balance
            + 0.01 * (60 - age)
        )
        prob = 1 / (1 + np.exp(-logits))
        default = (rng.rand(n) < prob).astype(int)
        return pd.DataFrame({
            'gender': gender, 'age': age, 'farm_size': farm_size, 'crop': crop,
            'cooperative': cooperative, 'yield_hist': yield_hist, 'mobile_txns': mobile_txns,
            'mobile_balance': mobile_balance, 'ndvi': ndvi, 'drought_exposure': drought_exposure,
            'default': default
        })

    df = generate_farmers()
    X, y = df.drop("default", axis=1), df["default"]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.legend()
    st.pyplot(fig)
