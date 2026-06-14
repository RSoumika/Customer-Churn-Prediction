import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1e1e2f;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white;
}

/* Main title */
h1 {
    color: white;
    font-weight: bold;
}

/* Predict button */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #3b82f6;
    color: white;
}

/* Remove Streamlit footer */
footer {
    visibility: hidden;
}

/* Hide menu */
#MainMenu {
    visibility: hidden;
}

/* Metric boxes */
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📊 Project Information")

st.sidebar.markdown("""
### Customer Churn Prediction

This application predicts whether a telecom customer is likely to churn using a machine learning model.

### Model Details

- Algorithm: Logistic Regression
- Recall: 79.41%
- F1 Score: 61.30%
- Accuracy: 73.35%

### Business Goal

Identify customers at risk of leaving and support retention strategies.
""")

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# =========================
# MAIN PAGE
# =========================

st.title("📞 Customer Churn Prediction System")

st.caption(
    "Predict whether a telecom customer is likely to leave the service based on contract type, tenure, and billing information."
)

st.markdown("---")

st.subheader("👤 Customer Information")

# =========================
# INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    contract = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

# =========================
# DATAFRAME
# =========================

input_data = pd.DataFrame(
    0,
    index=[0],
    columns=feature_names
)

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges

# Contract encoding

if contract == "One year":
    input_data["Contract_One year"] = 1

elif contract == "Two year":
    input_data["Contract_Two year"] = 1

# Internet encoding

if internet_service == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1

elif internet_service == "No":
    input_data["InternetService_No"] = 1

# =========================
# PREDICTION
# =========================

if st.button("🔍 Predict Churn"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction:

        st.error("⚠️ High Churn Risk")

        st.progress(float(probability))

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    else:

        st.success("✅ Low Churn Risk")

        st.progress(float(probability))

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Built by Soumika | Customer Churn Prediction using Machine Learning and Streamlit"
)