import streamlit as st
import joblib
import numpy as np

# Load the saved models
log_reg_model = joblib.load('logistic_regression_model.pkl')
svm_model = joblib.load('svm_model.pkl')
gb_model = joblib.load('gradient_boosting_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Define the Streamlit app
st.title("Cardio Prediction App")

st.write("""
This app predicts whether a person has cardiovascular disease based on their health metrics.
""")

# Input fields for user to enter data
gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
ap_hi = st.number_input("Systolic blood pressure (ap_hi)", min_value=50, max_value=200, value=120)
ap_lo = st.number_input("Diastolic blood pressure (ap_lo)", min_value=30, max_value=120, value=80)
cholesterol = st.selectbox("Cholesterol level", [1, 2, 3], format_func=lambda x: "Normal" if x == 1 else "Above Normal" if x == 2 else "Well Above Normal")
gluc = st.selectbox("Glucose level", [1, 2, 3], format_func=lambda x: "Normal" if x == 1 else "Above Normal" if x == 2 else "Well Above Normal")
smoke = st.selectbox("Do you smoke?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
alco = st.selectbox("Do you consume alcohol?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
active = st.selectbox("Are you physically active?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
age_years = st.number_input("Age in years", min_value=1, max_value=120, value=30)
BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
pulse_pressure = ap_hi - ap_lo
mean_arterial_pressure = (ap_hi + 2 * ap_lo) / 3

# One-hot encoding for cholesterol
chol_1 = 1 if cholesterol == 1 else 0
chol_2 = 1 if cholesterol == 2 else 0
chol_3 = 1 if cholesterol == 3 else 0

# One-hot encoding for glucose
gluc_1 = 1 if gluc == 1 else 0
gluc_2 = 1 if gluc == 2 else 0
gluc_3 = 1 if gluc == 3 else 0

# Create a numpy array from the input data
input_data = np.array([[gender, ap_hi, ap_lo, smoke, alco, active, age_years, BMI, 
                        chol_1, chol_2, chol_3, gluc_1, gluc_2, gluc_3, 
                        pulse_pressure, mean_arterial_pressure]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make predictions with the loaded models
log_reg_pred = log_reg_model.predict(input_data_scaled)
svm_pred = svm_model.predict(input_data_scaled)
gb_pred = gb_model.predict(input_data_scaled)

# Display the predictions
st.write(f"Logistic Regression Prediction: {'Cardio' if log_reg_pred[0] == 1 else 'No Cardio'}")
st.write(f"SVM Prediction: {'Cardio' if svm_pred[0] == 1 else 'No Cardio'}")
st.write(f"Gradient Boosting Classifier Prediction: {'Cardio' if gb_pred[0] == 1 else 'No Cardio'}")
