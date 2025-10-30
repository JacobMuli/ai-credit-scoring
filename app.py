# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

# --- Page setup ---
st.set_page_config(page_title="🌾 AI Credit Scoring", layout="wide")

# --- Load model ---
MODEL_PATH = "credit_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Please upload 'credit_model.pkl' to your repo.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("🌾 AI Credit Scoring for Smallholder Farmers")

# --- Top navigation tabs ---
tab_predict, tab_dashboard = st.tabs(["🧾 Predict Farmer Score", "📊 Model Dashboard"])

# =============================================================
# TAB 1: PREDICTION
# =============================================================
with tab_predict:
    st.write("Use the sidebar to fill in the farmer’s profile and get a credit score prediction.")

    with st.sidebar:
        st.markdown("### 📋 Farmer Profile")
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

    st.markdown("---")

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
            st.success("✅ Farmer is eligible for credit!")
            st.write(f"**Suggested Loan Amount:** KES {loan_amount:,.0f}")
            st.write(f"**Suggested Interest Rate:** {interest_rate*100:.2f}%")
        else:
            st.error("❌ Farmer not eligible for a loan at this time.")

# =============================================================
# TAB 2: DASHBOARD
# =============================================================
with tab_dashboard:
    st.subheader("📊 Model Performance Dashboard")
    st.write("Visualize how the model performs on synthetic test data (AUC, confusion matrix, and feature importance).")

    def generate_farmers(n=1000, seed=42):
        rng = np.random.RandomState(seed)
        gender = rng.choice(['Male', 'Female'], size=n, p=[0.6, 0.4])
        age = rng.randint(18, 70, size=n)
        farm_size = np.round(np.exp(rng.normal(np.log(2.0), 0.8, size=n)), 2)
        crop = rng.choice(['Maize', 'Beans', 'Tea', 'Coffee', 'Horticulture'], size=n, p=[0.4,0.25,0.15,0.1,0.1])
        cooperative = rng.binomial(1, p=0.4, size=n)
        yield_hist = np.maximum(0.1, rng.normal(2.0, 0.8, size=n))
        mobile_txns = rng.poisson(25, size=n)
        mobile_balance = np.maximum(0, rng.normal(1200, 600, size=n))
        ndvi = np.clip(rng.normal(0.45 + 0.1*(yield_hist/3.0), 0.08), 0.05, 0.9)
        drought_exposure = rng.binomial(1, p=0.3, size=n)
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
            'gender': gender,
            'age': age,
            'farm_size': farm_size,
            'crop': crop,
            'cooperative': cooperative,
            'yield_hist': yield_hist,
            'mobile_txns': mobile_txns,
            'mobile_balance': mobile_balance,
            'ndvi': ndvi,
            'drought_exposure': drought_exposure,
            'default': default
        })

    data = generate_farmers()
    X = data.drop(columns=["default"])
    y_true = data["default"]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    # --- Feature Importance ---
st.subheader("🌿 Top 10 Most Important Features")
try:
    preprocessor = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    # Extract transformed feature names safely
    # Works for sklearn >= 1.0
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        # Fallback for older sklearn versions
        num_features = ["age","farm_size","yield_hist","mobile_txns","mobile_balance","ndvi","cooperative","drought_exposure"]
        cat_features = ["gender","crop"]
        ohe = preprocessor.named_transformers_["cat"]
        ohe_names = list(ohe.get_feature_names_out(cat_features))
        feature_names = num_features + ohe_names

    importances = clf.feature_importances_

    # Ensure equal length
    if len(importances) == len(feature_names):
        feat_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=feat_imp.head(10), x="Importance", y="Feature", ax=ax)
        ax.set_title("Top 10 Most Important Features")
        st.pyplot(fig)
    else:
        st.warning(f"⚠️ Feature importance mismatch: expected {len(importances)}, got {len(feature_names)} names.")
except Exception as e:
    st.error(f"❌ Could not display feature importance: {e}")

