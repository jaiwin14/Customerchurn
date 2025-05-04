import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open('gradient_boost_model.sav', 'rb') as file:
    model = pickle.load(file)

# Load model columns
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📉 Telco Customer Churn Prediction App")

# Input form
def user_input_features():
    gender = st.selectbox("Gender", ['Male', 'Female'])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ['Yes', 'No'])
    Dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.number_input("Tenure (months)", min_value=0)
    PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
    MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
    PaymentMethod = st.selectbox("Payment Method", [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    return pd.DataFrame([data])

input_df = user_input_features()

if st.button("🔍 Predict Churn"):
    if not set(model_columns).issubset(input_df.columns):
        st.error("Input features mismatch. Please check input column names.")
    else:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        result = "⚠️ Churn" if prediction == 1 else "✅ No Churn"
        st.subheader(f"Prediction: {result}")
        st.write(f"Probability of Churn: {proba:.2%}")
