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

    # Default placeholder values to avoid NameError
    new_score = None

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
        dynamic = dynamic.apply(pd.to_numeric, errors="ignore")

        # Apply parameter changes safely
        dynamic["ndvi"] = np.clip(dynamic["ndvi"] + delta_ndvi, 0.05, 0.9)
        dynamic["mobile_txns"] = np.maximum(dynamic["mobile_txns"] + delta_txns, 0)
        dynamic["yield_hist"] = np.maximum(dynamic["yield_hist"] + delta_yield, 0.1)

        # Apply scenario shocks
        if drought_shock:
            dynamic["drought_exposure"] = 1
        if price_drop:
            dynamic["mobile_balance"] = dynamic["mobile_balance"] * 0.8
        if repayment_delay:
            dynamic["cooperative"] = 0

        # Predict updated risk
        new_score, new_prob, eligible, loan_amount, interest_rate = predict_credit(model, dynamic)

        # Save results for persistence
        st.session_state["new_score"] = float(new_score)
        st.session_state["new_prob"] = float(new_prob)

        # Display updated metrics
        st.metric("New Credit Score", f"{new_score:.0f}", delta=f"{new_score - st.session_state.get('base_score', new_score):.1f}")
        st.metric("New Default Probability", f"{new_prob:.2%}", delta=f"{(new_prob - st.session_state.get('base_prob', new_prob))*100:.2f}%")

        if eligible:
            st.success("✅ Still eligible for credit.")
        else:
            st.warning("⚠️ Farmer now ineligible due to increased risk.")

    # ---- Visualization (only render if simulation ran) ----
    st.markdown("#### 📈 Credit Score Comparison")

    base_score = st.session_state.get("base_score")
    updated_score = st.session_state.get("new_score")

    if base_score is not None and updated_score is not None:
        base_score = float(np.ravel(base_score)[0]) if hasattr(base_score, "__len__") else float(base_score)
        updated_score = float(np.ravel(updated_score)[0]) if hasattr(updated_score, "__len__") else float(updated_score)

        fig, ax = plt.subplots()
        ax.bar(["Base", "Updated"], [base_score, updated_score], color=["#4CAF50", "#FF9800"])
        ax.set_ylabel("Credit Score")
        ax.set_ylim(0, 1000)
        ax.set_title("Credit Score Before vs After Simulation")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Run the credit prediction and simulation to view the comparison chart.")


    # =====================================================
    # 💰 Repayment Schedule Simulator
    # =====================================================
    st.markdown("### 💰 Repayment Schedule Simulator")
    st.write(
        "Simulate how the suggested loan amount would be repaid monthly "
        "based on the predicted interest rate."
    )

    # Only run if a loan has been computed from either prediction or simulation
    loan_amount = st.session_state.get("loan_amount", None)
    interest_rate = st.session_state.get("interest_rate", None)

    # If simulation ran, reuse latest; otherwise, prompt user
    if "new_score" in st.session_state and "new_prob" in st.session_state:
        # Estimate loan and rate again from latest dynamic sample
        est_rate = 0.12 + st.session_state["new_prob"] * 0.5
        est_amount = min(sample["farm_size"].values[0] * 300, 50000)
    elif "base_prob" in st.session_state:
        est_rate = 0.12 + st.session_state["base_prob"] * 0.5
        est_amount = min(sample["farm_size"].values[0] * 300, 50000)
    else:
        est_rate = None
        est_amount = None

    if est_rate is not None and est_amount is not None:
        colA, colB = st.columns(2)
        with colA:
            months = st.slider("Repayment period (months)", 3, 24, 12)
        with colB:
            rate = st.number_input(
                "Annual interest rate (%)", 1.0, 60.0, round(est_rate * 100, 2)
            )

        # Compute monthly payment using annuity formula
        r = rate / 100 / 12
        n = months
        P = est_amount
        monthly_payment = (P * r) / (1 - (1 + r) ** -n)

        # Build amortization schedule
        schedule = []
        balance = P
        for i in range(1, n + 1):
            interest = balance * r
            principal = monthly_payment - interest
            balance -= principal
            schedule.append(
                {
                    "Month": i,
                    "Principal": round(principal, 2),
                    "Interest": round(interest, 2),
                    "Payment": round(monthly_payment, 2),
                    "Remaining Balance": round(max(balance, 0), 2),
                }
            )

        df = pd.DataFrame(schedule)

        st.write(f"**Loan Amount:** KES {P:,.0f}")
        st.write(f"**Monthly Payment:** KES {monthly_payment:,.2f}")
        st.write(f"**Total Payment:** KES {df['Payment'].sum():,.2f}")
        st.write(f"**Total Interest:** KES {df['Interest'].sum():,.2f}")

        # Show table and plot
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("#### 📊 Repayment Progress")
        fig, ax = plt.subplots()
        ax.plot(df["Month"], df["Remaining Balance"], marker="o")
        ax.set_xlabel("Month")
        ax.set_ylabel("Remaining Balance (KES)")
        ax.set_title("Loan Amortization Curve")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Run a credit prediction or simulation first to generate a loan estimate.")


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
