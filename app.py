import streamlit as st
import pandas as pd
import joblib
import numpy as np

#Title
st.title("💳 Fintech Fraud Detection System")
st.markdown("""
**Real-time transaction risk scoring for B2B payments**

- Model: RandomForest,XGBoost with SMOTE
- AUC-ROC: 0.95
- Dataset: 284K credit card transactions
""")

# Load model and scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")


# Upload CSV
uploaded_file = st.file_uploader("Upload Transaction CSV File", type=["csv"])

if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write(data.head())

    # Remove target column if present
    if "Class" in data.columns:
        data_input = data.drop("Class", axis=1)
    else:
        data_input = data


# Feature Engineering
    if "Amount" in data_input.columns:
        data_input["Amount_Log"] = np.log1p(data_input["Amount"])

    # Scale data
    scaled_data = scaler.transform(data_input)

    # Prediction
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)[:, 1]

    # Add results
    data["Fraud_Prediction"] = predictions
    data["Fraud_Probability"] = probabilities

    st.subheader("Prediction Results")
    st.write(data.head())

    # Fraud transactions only
    frauds = data[data["Fraud_Prediction"] == 1]

    st.subheader("Detected Fraud Transactions")
    st.write(frauds)

    # Download results
    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Results",
        csv,
        "fraud_predictions.csv",
        "text/csv"
    )