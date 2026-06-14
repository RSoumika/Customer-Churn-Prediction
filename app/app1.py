import streamlit as st
import pandas as pd
import joblib
import os

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



BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "churn_model.pkl")
)

feature_names = joblib.load(
    os.path.join(BASE_DIR, "models", "feature_names.pkl")
)

# =========================
# MAIN PAGE
# =========================

st.title("📞 Customer Churn Prediction System")

st.caption(
    "Predict whether a telecom customer is likely to leave the service based on contract type, tenure, and billing information."
)

st.markdown("---")

left, center, right = st.columns([1, 3, 1])

with center:

    def yes_no(label):
        return st.selectbox(label, ["No", "Yes"])

    with st.expander("👤 Customer Information"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

        with col2:
            partner = yes_no("Partner")
            dependents = yes_no("Dependents")

    with st.expander("📞 Phone Services"):

        col1, col2 = st.columns(2)

        with col1:
            phone_service = yes_no("Phone Service")

        with col2:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "No",
                    "Yes",
                    "No phone service"
                ]
            )

    with st.expander("🌐 Internet Services"):

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        col1, col2 = st.columns(2)

        with col1:

            online_security = st.selectbox(
                "Online Security",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            online_backup = st.selectbox(
                "Online Backup",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            device_protection = st.selectbox(
                "Device Protection",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

        with col2:

            tech_support = st.selectbox(
                "Tech Support",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

    with st.expander("💳 Billing Information"):

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = yes_no("Paperless Billing")

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

        total_charges = monthly_charges * tenure

# =========================
# CREATE MODEL INPUT
# =========================

input_data = pd.DataFrame(
    0,
    index=[0],
    columns=feature_names
)

# Numeric Features
input_data["SeniorCitizen"] = senior
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = total_charges

# Gender
if gender == "Male":
    input_data["gender_Male"] = 1

# Partner
if partner == "Yes":
    input_data["Partner_Yes"] = 1

# Dependents
if dependents == "Yes":
    input_data["Dependents_Yes"] = 1

# Phone Service
if phone_service == "Yes":
    input_data["PhoneService_Yes"] = 1

# Multiple Lines
if multiple_lines == "Yes":
    input_data["MultipleLines_Yes"] = 1
elif multiple_lines == "No phone service":
    input_data["MultipleLines_No phone service"] = 1

# Internet Service
if internet_service == "Fiber optic":
    input_data["InternetService_Fiber optic"] = 1
elif internet_service == "No":
    input_data["InternetService_No"] = 1

# Online Security
if online_security == "Yes":
    input_data["OnlineSecurity_Yes"] = 1
elif online_security == "No internet service":
    input_data["OnlineSecurity_No internet service"] = 1

# Online Backup
if online_backup == "Yes":
    input_data["OnlineBackup_Yes"] = 1
elif online_backup == "No internet service":
    input_data["OnlineBackup_No internet service"] = 1

# Device Protection
if device_protection == "Yes":
    input_data["DeviceProtection_Yes"] = 1
elif device_protection == "No internet service":
    input_data["DeviceProtection_No internet service"] = 1

# Tech Support
if tech_support == "Yes":
    input_data["TechSupport_Yes"] = 1
elif tech_support == "No internet service":
    input_data["TechSupport_No internet service"] = 1

# Streaming TV
if streaming_tv == "Yes":
    input_data["StreamingTV_Yes"] = 1
elif streaming_tv == "No internet service":
    input_data["StreamingTV_No internet service"] = 1

# Streaming Movies
if streaming_movies == "Yes":
    input_data["StreamingMovies_Yes"] = 1
elif streaming_movies == "No internet service":
    input_data["StreamingMovies_No internet service"] = 1

# Contract
if contract == "One year":
    input_data["Contract_One year"] = 1
elif contract == "Two year":
    input_data["Contract_Two year"] = 1

# Paperless Billing
if paperless == "Yes":
    input_data["PaperlessBilling_Yes"] = 1

# Payment Method
if payment_method == "Credit card (automatic)":
    input_data["PaymentMethod_Credit card (automatic)"] = 1

elif payment_method == "Electronic check":
    input_data["PaymentMethod_Electronic check"] = 1

elif payment_method == "Mailed check":
    input_data["PaymentMethod_Mailed check"] = 1

# =========================
# PREDICT BUTTON
# =========================

if st.button("🔍 Predict Churn", use_container_width=True):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    if prediction:

        st.error(
            f"⚠️ High Churn Risk ({probability:.2%})"
        )

    else:

        st.success(
            f"✅ Low Churn Risk ({probability:.2%})"
        )

    st.progress(float(probability))

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )
    if probability > 0.8:
        st.error("🚨 Immediate retention action recommended.")

    elif probability > 0.5:
        st.warning("⚠ Customer should be monitored closely.")

    else:
        st.success("✅ Customer appears stable.")

    st.subheader("📌 Key Risk Factors")

    risks = []

    if tenure < 12:
        risks.append("Low customer tenure")

    if contract == "Month-to-month":
        risks.append("Month-to-month contract")

    if monthly_charges > 80:
        risks.append("High monthly charges")

    if tech_support == "No":
        risks.append("No tech support service")

    if internet_service == "Fiber optic":
        risks.append("Fiber optic customers historically show higher churn")

    if len(risks) > 0:

        for risk in risks:
            st.warning(risk)

    else:
        st.write("No major churn indicators detected.")
# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Built by Soumika | Customer Churn Prediction using Machine Learning and Streamlit"
)