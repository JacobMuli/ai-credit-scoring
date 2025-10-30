# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="📊 Model Dashboard", layout="wide")
st.title("📊 Credit Scoring Model Dashboard")

MODEL_PATH = "credit_model.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error("Model file not found. Please ensure credit_model.pkl is in the project root.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()
if model is None:
    st.stop()

st.markdown("""
### 🌾 Model Overview
This dashboard provides insights into the trained AI Credit Scoring model, including:
- ROC Curve  
- Confusion Matrix  
- Feature Importance  
- Classification Metrics
""")

# Generate synthetic test data to evaluate model
st.markdown("---")
st.subheader("🧪 Model Evaluation (using synthetic data)")

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
X = data.drop(columns=['default'])
y_true = data['default']
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# --- ROC Curve ---
st.subheader("🎯 ROC Curve")
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = roc_auc_score(y_true, y_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
st.pyplot(fig)

# --- Confusion Matrix ---
st.subheader("🧩 Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
st.pyplot(fig)

# --- Classification Report ---
st.subheader("📋 Classification Report")
report = classification_report(y_true, y_pred, target_names=["No Default", "Default"], output_dict=True)
st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap="YlGnBu"))

# --- Feature Importance ---
st.subheader("🌿 Feature Importance")
try:
    ohe = model.named_steps["pre"].named_transformers_["cat"]
    cat_features = ['gender', 'crop']
    ohe_names = list(ohe.get_feature_names_out(cat_features))
    num_features = ['age','farm_size','yield_hist','mobile_txns','mobile_balance','ndvi','cooperative','drought_exposure']
    feature_names = num_features + ohe_names

    importances = model.named_steps['clf'].feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_imp.head(10), x='Importance', y='Feature', ax=ax)
    ax.set_title("Top 10 Most Important Features")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"⚠️ Could not display feature importance: {e}")
