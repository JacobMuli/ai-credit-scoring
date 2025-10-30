# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, gzip, requests, io, os
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# -----------------------------------------------------
# 🌾 PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="🌾 AI Credit Scoring", layout="wide")

st.title("🌾 AI Credit Scoring for Smallholder Farmers")
st.write("A predictive simulation to assess farmers’ creditworthiness and simulate real-world lending outcomes.")

# -----------------------------------------------------
# 📦 LOAD MODEL (LOCAL + GITHUB FALLBACK)
# -----------------------------------------------------
MODEL_PATH = "credit_model.pkl.gz"
GITHUB_REPO_URL = "https://github.com/JacobMuli/ai-credit-scoring/raw/main/credit_model.pkl.gz"

@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            with gzip.open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            return model, "✅ Model loaded locally."
        else:
            st.info("📥 Downloading model from GitHub...")
            response = requests.get(GITHUB_REPO_URL)
            if response.status_code == 200:
                model = pickle.load(io.BytesIO(response.content))
                return model, "✅ Model loaded from GitHub successfully!"
            else:
                return None, "❌ Model not found. Check GitHub link or local path."
    except Exception as e:
        return None, f"⚠️ Error loading model: {e}"

model, model_status = load_model()
if not model:
    st.error(model_status)
    st.stop()
else:
    st.success(model_status)

# -----------------------------------------------------
# 🔖 TABS
# -----------------------------------------------------
tab_predict, tab_simulation, tab_dashboard = st.tabs([
    "🧾 Predict Farmer Score",
    "⚙️ Risk Simulation & What-If Testing",
    "📊 Model Dashboard"
])

# =====================================================
# TAB 1: FARMER PREDICTION
# =====================================================
with tab_predict:
    st.sidebar.header("📋 Farmer Profile")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 18, 80, 35)
    farm_size = st.sidebar.number_input("Farm size (hectares)", 0.1, 100.0, 3.5)
    crop = st.sidebar.selectbox("Main Crop", ["Maize", "Beans", "Tea", "Coffee", "Horticulture"])
    cooperative = st.sidebar.selectbox("Member of Cooperative", [0, 1])
    yield_hist = st.sidebar.number_input("Average yield (tons/ha)", 0.1, 10.0, 2.5)
    mobile_txns = st.sidebar.number_input("Monthly Mobile Transactions", 0, 200, 25)
    mobile_balance = st.sidebar.number_input("Avg. Mobile Wallet Balance (KES)", 0, 100000, 1500)
    ndvi = st.sidebar.slider("NDVI (Vegetation Health)", 0.05, 0.9, 0.55)
    drought_exposure = st.sidebar.selectbox("Drought Exposure (recent)", [0, 1])

    sample = pd.DataFrame([{
        "gender": gender, "age": age, "farm_size": farm_size, "crop": crop,
        "cooperative": cooperative, "yield_hist": yield_hist,
        "mobile_txns": mobile_txns, "mobile_balance": mobile_balance,
        "ndvi": ndvi, "drought_exposure": drought_exposure
    }])

    def predict_credit(model, data):
        prob_default = model.predict_proba(data)[0, 1]
        credit_score = (1 - prob_default) * 1000
        eligible = credit_score >= 400
        loan_amount = min(data["farm_size"].values[0] * 300, 50000)
        interest_rate = 0.12 + prob_default * 0.5
        return credit_score, prob_default, eligible, loan_amount, interest_rate

    if st.button("🚀 Predict Credit Score"):
        credit_score, prob_default, eligible, loan_amount, interest_rate = predict_credit(model, sample)
        st.subheader("🔍 Prediction Results")
        st.metric("Credit Score", f"{credit_score:.0f}")
        st.metric("Default Probability", f"{prob_default:.2%}")
        if eligible:
            st.success("✅ Farmer is eligible for credit!")
            st.write(f"**Suggested Loan Amount:** KES {loan_amount:,.0f}")
            st.write(f"**Suggested Interest Rate:** {interest_rate*100:.2f}%")
        else:
            st.error("❌ Farmer not eligible for a loan at this time.")
        st.session_state["base_score"] = credit_score
        st.session_state["base_prob"] = prob_default

# =====================================================
# TAB 2: DYNAMIC RISK SIMULATION
# =====================================================
with tab_simulation:
    st.subheader("⚙️ Dynamic Risk & What-If Simulation")
    st.write("Simulate how environmental or behavioral changes affect the farmer’s credit score and eligibility.")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Adjust Farmer Parameters:")
        delta_ndvi = st.slider("NDVI Change (Δ)", -0.3, 0.3, 0.0, step=0.01)
        delta_txns = st.slider("Change in Mobile Transactions (Δ)", -50, 50, 0, step=5)
        delta_yield = st.slider("Change in Yield (Δ tons/ha)", -2.0, 2.0, 0.0, step=0.1)
    with col2:
        st.write("Simulate External Shocks:")
        drought_shock = st.checkbox("🌵 Apply Drought Shock?")
        price_drop = st.checkbox("📉 Market Price Drop?")
        repayment_delay = st.checkbox("⌛ Repayment Delay Risk?")

    if st.button("🔁 Run Simulation"):
        base = sample.copy()
        dynamic = base.copy()
        dynamic["ndvi"] = np.clip(dynamic["ndvi"] + delta_ndvi, 0.05, 0.9)
        dynamic["mobile_txns"] = max(dynamic["mobile_txns"] + delta_txns, 0)
        dynamic["yield_hist"] = max(dynamic["yield_hist"] + delta_yield, 0.1)

        if drought_shock: dynamic["drought_exposure"] = 1
        if price_drop: dynamic["mobile_balance"] *= 0.8
        if repayment_delay: dynamic["cooperative"] = 0

        new_score, new_prob, eligible, loan_amount, interest_rate = predict_credit(model, dynamic)

        st.metric("New Credit Score", f"{new_score:.0f}", delta=f"{new_score - st.session_state.get('base_score', new_score):.1f}")
        st.metric("New Default Probability", f"{new_prob:.2%}", delta=f"{(new_prob - st.session_state.get('base_prob', new_prob))*100:.2f}%")

        if eligible:
            st.success("✅ Still eligible for credit.")
        else:
            st.warning("⚠️ Farmer now ineligible due to increased risk.")

        st.markdown("#### 📈 Credit Score Comparison")
        fig, ax = plt.subplots()
        ax.bar(["Base", "Updated"], [st.session_state.get("base_score", new_score), new_score], color=["#4CAF50", "#FF9800"])
        ax.set_ylabel("Credit Score")
        ax.set_ylim(0, 1000)
        st.pyplot(fig)

# =====================================================
# TAB 3: MODEL DASHBOARD
# =====================================================
with tab_dashboard:
    st.subheader("📊 Model Performance Dashboard")
    st.write("Evaluate model accuracy using synthetic farmer data.")

    def generate_farmers(n=500):
        rng = np.random.default_rng(42)
        gender = rng.choice(['Male', 'Female'], n)
        age = rng.integers(18, 70, n)
        farm_size = np.round(np.exp(rng.normal(np.log(2.0), 0.8, n)), 2)
        crop = rng.choice(['Maize', 'Beans', 'Tea', 'Coffee', 'Horticulture'], n)
        cooperative = rng.integers(0, 2, n)
        yield_hist = np.maximum(0.1, rng.normal(2.0, 0.8, n))
        mobile_txns = rng.poisson(25, n)
        mobile_balance = np.maximum(0, rng.normal(1200, 600, n))
        ndvi = np.clip(rng.normal(0.45, 0.1, n), 0.05, 0.9)
        drought_exposure = rng.integers(0, 2, n)
        logits = (-1.0*np.log(farm_size+0.1) - 0.8*cooperative - 1.2*(yield_hist/np.maximum(farm_size,0.1))
                  + 1.5*drought_exposure - 0.001*mobile_balance + 0.01*(60-age))
        prob = 1 / (1 + np.exp(-logits))
        default = (rng.random(n) < prob).astype(int)
        return pd.DataFrame({
            "gender": gender, "age": age, "farm_size": farm_size, "crop": crop,
            "cooperative": cooperative, "yield_hist": yield_hist,
            "mobile_txns": mobile_txns, "mobile_balance": mobile_balance,
            "ndvi": ndvi, "drought_exposure": drought_exposure, "default": default
        })

    data = generate_farmers()
    X = data.drop(columns="default")
    y_true = data["default"]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.legend(); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    st.pyplot(fig)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)

    # --- Feature Importance ---
    st.subheader("🌿 Top 10 Most Important Features")
    try:
        pre = model.named_steps["pre"]; clf = model.named_steps["clf"]
        try:
            feat_names = pre.get_feature_names_out()
        except:
            num_feats = ["age","farm_size","yield_hist","mobile_txns","mobile_balance","ndvi","cooperative","drought_exposure"]
            cat_feats = ["gender","crop"]
            ohe = pre.named_transformers_["cat"]
            ohe_names = list(ohe.get_feature_names_out(cat_feats))
            feat_names = num_feats + ohe_names
        importances = clf.feature_importances_
        feat_imp = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(data=feat_imp.head(10), x="Importance", y="Feature", ax=ax)
        ax.set_title("Top 10 Features Driving Predictions")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")
