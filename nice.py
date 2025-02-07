import streamlit as st
import joblib
import numpy as np

# Load the saved Gradient Boosting model
gb_model = joblib.load('gradient_boosting_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Define the Streamlit app
st.set_page_config(page_title="Cardiovascular Disease Prediction App", layout="wide")
st.title("Cardiovascular Disease Prediction App")

st.sidebar.header("About")
st.sidebar.write("""
This app predicts whether a person has cardiovascular disease based on their health metrics.
Please fill in the details below to get the prediction from the Gradient Boosting model.
""")

# Input fields for user to enter data
st.header("Enter Your Health Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    age_years = st.slider("Age in years", min_value=1, max_value=120, value=30)
    smoke = st.selectbox("Do you smoke?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    alco = st.selectbox("Do you consume alcohol?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    active = st.selectbox("Are you physically active?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    ap_hi = st.slider("Systolic blood pressure (ap_hi)", min_value=50, max_value=200, value=120)
    ap_lo = st.slider("Diastolic blood pressure (ap_lo)", min_value=30, max_value=120, value=80)
    cholesterol = st.selectbox("Cholesterol level", [1, 2, 3], 
                                format_func=lambda x: "Normal" if x == 1 else "Above Normal" if x == 2 else "Well Above Normal")
    gluc = st.selectbox("Glucose level", [1, 2, 3], 
                         format_func=lambda x: "Normal" if x == 1 else "Above Normal" if x == 2 else "Well Above Normal")

with col3:
    BMI = st.slider("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0)
    pulse_pressure = ap_hi - ap_lo
    mean_arterial_pressure = (ap_hi + 2 * ap_lo) / 3
    st.write(f"Pulse Pressure: {pulse_pressure}")
    st.write(f"Mean Arterial Pressure: {mean_arterial_pressure:.2f}")

# One-hot encoding for cholesterol and glucose
chol_1, chol_2, chol_3 = (cholesterol == i for i in range(1, 4))
gluc_1, gluc_2, gluc_3 = (gluc == i for i in range(1, 4))

# Create a numpy array from the input data
input_data = np.array([[gender, ap_hi, ap_lo, smoke, alco, active, age_years, BMI, 
                        chol_1, chol_2, chol_3, gluc_1, gluc_2, gluc_3, 
                        pulse_pressure, mean_arterial_pressure]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make predictions with the Gradient Boosting model
gb_pred = gb_model.predict(input_data_scaled)

# Display the prediction
st.header("Prediction")
st.subheader("Gradient Boosting")
st.success("No Cardiovascular Disease predicted" if gb_pred[0] == 0 else "Cardiovascular Disease predicted")

st.sidebar.write("For more information, contact the developer.")
