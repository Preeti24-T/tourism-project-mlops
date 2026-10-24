
# ============================================================
# STREAMLIT APPLICATION
# Wellness Tourism Purchase Prediction
# ============================================================

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# 1. Configure application
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="✈️"
)

st.title("Wellness Tourism Purchase Prediction")

st.write(
    "Predict customer purchase likelihood using "
    "information available before sales contact."
)

# ------------------------------------------------------------
# 2. Load trained model and feature schema
# ------------------------------------------------------------
model_path = ROOT / "models/best_model.joblib"
schema_path = ROOT / "models/feature_schema.json"

if not model_path.exists():
    st.error("Trained model not found.")
    st.stop()

if not schema_path.exists():
    st.error("Feature schema not found.")
    st.stop()

model = joblib.load(model_path)

with open(schema_path, "r") as file:
    feature_schema = json.load(file)

expected_columns = feature_schema["columns"]

# ------------------------------------------------------------
# 3. Create customer input form
# ------------------------------------------------------------
with st.form("prediction_form"):

    st.subheader("Customer Information")

    # CustomerID is used only for identification.
    customer_id = st.text_input("Customer ID")

    # Numerical customer information.
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    city = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    # Categorical customer information.
    occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Freelancer"
        ]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    visitors = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2
    )

    property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    marital = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    trips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=30,
        value=3
    )

    passport = st.selectbox(
        "Passport",
        [0, 1]
    )

    car = st.selectbox(
        "Own Car",
        [0, 1]
    )

    children = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=15,
        value=0
    )

    designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    income = st.number_input(
        "Monthly Income",
        min_value=0,
        max_value=1000000,
        value=25000
    )

    submitted = st.form_submit_button(
        "Predict Purchase"
    )

# ------------------------------------------------------------
# 4. Generate customer prediction
# ------------------------------------------------------------
if submitted:

    # Create the same 13 predictor columns used in training.
    customer = pd.DataFrame([{
        "Age": age,
        "CityTier": city,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": visitors,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "OwnCar": car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income
    }])

    # Validate input columns against the saved training schema.
    missing_columns = (
        set(expected_columns) - set(customer.columns)
    )

    extra_columns = (
        set(customer.columns) - set(expected_columns)
    )

    if missing_columns or extra_columns:
        st.error(
            f"Feature mismatch. "
            f"Missing: {missing_columns}. "
            f"Extra: {extra_columns}."
        )
        st.stop()

    # Reorder inputs to match the training dataset.
    customer = customer[expected_columns]

    # Generate predicted class and purchase probability.
    prediction = int(model.predict(customer)[0])

    probability = float(
        model.predict_proba(customer)[0, 1]
    )

    # --------------------------------------------------------
    # 5. Display prediction results
    # --------------------------------------------------------
    st.write(
        "Customer ID:",
        customer_id or "Not provided"
    )

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.success(
            "Predicted: Likely to purchase"
        )
    else:
        st.warning(
            "Predicted: Unlikely to purchase"
        )

    st.caption(
        "This prediction uses pre-contact customer "
        "attributes and historical purchase patterns."
    )
