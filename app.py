# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="🌾 AI Credit Scoring", layout="wide")

st.title("🌾 AI Credit Scoring for Smallholder Farmers")
st.write("Predict and explain a farmer’s creditworthiness using AI models trained on agricultural, financial, and remote-sensing data.")

MODEL_PATH_PKL = "credit_model.pkl"
MODEL_PATH_JOBLIB = "credit_model.joblib"

# Try to load model safely
@st.cache_resource
def load_model():
    model = None
    if os.path.exists(MODEL_PATH_PKL):
        with open(MODEL_PATH_PKL, "rb") as f:
            model = pickle.load(f)
    elif os.path.exists(MODEL_PATH_JOBLIB):
        model = joblib.load(MODEL_PATH_JOBLIB)
    return model

model = load_model()

if model is None:
    st.error("❌ No model file found. Please upload `credit_model.pkl` or `credit_model.joblib` to the repo.")
    st.stop()

# Sidebar inputs
with st.sidebar:
    st.markdown("### 📋 Farmer Profile")
    st.write("Fill in the farmer’s details below:")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 80, 35)
    farm_size = st.number_input("Farm size (hectares)", 0.1, 100.0, 3.5, step=0.1)
    crop = st.selectbox("Main Crop", ["Maize", "Beans", "Tea", "Coffee", "Horticulture"])
    cooperative = st.selectbox("Member of Cooperative", [0, 1])
    yield_hist = st.number_input("Average yield (tons/ha)", 0.1, 10.0, 2.5, step=0.1)
    mobile_txns = st.number_input("Monthly Mobile Transactions", 0, 200, 25)
    mobile_balance = st.number_input("Avg. Mobile Wallet Balance (KES)", 0, 100000, 1500)
    ndvi = st.slider("NDVI (Vegetation Health)", 0.05, 0.9, 0.55, step=0.01)
    drought_exposure = st.selectbox("Drought Exposure (recent)", [0, 1])

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

# Predict button
if st.button("🚀 Predict Credit Score"):
    prob_default = model.predict_proba(sample)[0, 1]
    credit_score = (1 - prob_default) * 1000
    eligible = credit_score >= 400

    st.subheader("🔍 Prediction Results")
    st.metric("Credit Score", f"{credit_score:.0f}")
    st.metric("Default Probability", f"{prob_default:.2%}")

    if eligible:
        loan_amount = min(sample["farm_size"].values[0] * 300, 50000)
        interest_rate = 0.12 + prob_default * 0.5
        st.success("✅ Farmer is **eligible** for credit!")
        st.write(f"**Suggested Loan Amount:** KES {loan_amount:,.0f}")
        st.write(f"**Suggested Interest Rate:** {interest_rate*100:.2f}%")
    else:
        st.error("❌ Farmer **not eligible** for a loan at this time.")

    st.markdown("---")

    # SHAP & LIME explanations section
    with st.expander("🧠 Explain Model Prediction"):
        st.write("Understanding **why** the model made this prediction helps build trust and transparency.")

        # ---- SHAP ----
        st.subheader("🔍 SHAP Global Feature Importance")

        # Generate background data
        explainer = shap.TreeExplainer(model.named_steps["clf"])
        X_transformed = model.named_steps["pre"].transform(sample)
        shap_values = explainer.shap_values(X_transformed)

        # Plot SHAP feature importance
        st.write("**Global Feature Importance:**")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_transformed, show=False)
        st.pyplot(fig)

        # ---- LIME ----
        st.subheader("🧩 LIME Local Explanation")

        # Prepare Lime explainer (fit on numeric-encoded data)
        preprocessor = model.named_steps["pre"]
        X_bg = preprocessor.transform(pd.DataFrame([{
            "gender": "Male",
            "age": 40,
            "farm_size": 2.0,
            "crop": "Maize",
            "cooperative": 1,
            "yield_hist": 2.5,
            "mobile_txns": 20,
            "mobile_balance": 1000,
            "ndvi": 0.45,
            "drought_exposure": 0
        } for _ in range(100)]))

        lime_explainer = LimeTabularExplainer(
            training_data=X_bg,
            feature_names=model.named_steps["pre"].get_feature_names_out(),
            mode="classification"
        )

        instance = preprocessor.transform(sample)
        lime_exp = lime_explainer.explain_instance(instance[0], model.named_steps["clf"].predict_proba)

        # Show LIME explanation
        st.write("**Local Explanation (Top Factors):**")
        st.components.v1.html(lime_exp.as_html(), height=800, scrolling=True)
