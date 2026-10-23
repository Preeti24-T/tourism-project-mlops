from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="✈️"
)

st.title("Wellness Tourism Purchase Prediction")
st.write("XGBoost-based customer purchase prediction")

# Load saved XGBoost pipeline
model_path = ROOT / "models/best_model.joblib"

if not model_path.exists():
    st.error("Trained model not found.")
    st.stop()

model = joblib.load(model_path)

with st.form("prediction_form"):

    # CustomerID is retained for identification only
    customer_id = st.text_input("Customer ID")

    age = st.number_input("Age", 18, 100, 35)

    contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    city = st.selectbox("City Tier", [1, 2, 3])

    duration = st.number_input(
        "Duration of Pitch", 0, 120, 15
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business",
         "Large Business", "Freelancer"]
    )

    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

    visitors = st.number_input(
        "Number of Persons Visiting", 1, 20, 2
    )

    followups = st.number_input(
        "Number of Followups", 0, 20, 3
    )

    product = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard",
         "Super Deluxe", "King"]
    )

    property_star = st.selectbox(
        "Preferred Property Star", [3, 4, 5]
    )

    marital = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    trips = st.number_input(
        "Number of Trips", 0, 30, 3
    )

    passport = st.selectbox("Passport", [0, 1])

    satisfaction = st.slider(
        "Pitch Satisfaction Score", 1, 5, 3
    )

    car = st.selectbox("Own Car", [0, 1])

    children = st.number_input(
        "Number of Children Visiting", 0, 15, 0
    )

    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager",
         "Senior Manager", "AVP", "VP"]
    )

    income = st.number_input(
        "Monthly Income", 0, 1000000, 25000
    )

    submitted = st.form_submit_button(
        "Predict Purchase"
    )

if submitted:

    customer = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": visitors,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "PitchSatisfactionScore": satisfaction,
        "OwnCar": car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income
    }])

    # CustomerID is not passed to the model
    prediction = int(model.predict(customer)[0])

    probability = float(
        model.predict_proba(customer)[0, 1]
    )

    st.write("Customer ID:", customer_id or "Not provided")

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.success("Predicted: Likely to purchase")
    else:
        st.warning("Predicted: Unlikely to purchase")

    st.caption(
        "Prediction is based on historical data. "
        "Pitch-related inputs may only be available "
        "after customer contact."
    )
