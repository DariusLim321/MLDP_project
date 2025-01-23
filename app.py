import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

# Set page config as the first command
st.set_page_config(page_title="Cardiovascular Disease Prediction App", layout="wide")

# Function to encode the image in base64
def get_base64_image(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# Path to the image in your repository
# image_path = "cardio.jpg"  # Ensure this matches your uploaded image's name

# # Encode the image
# base64_image = get_base64_image(image_path)

# CSS for background image with overlay
overlay_css = f'''
<style>
.stApp {{
    background-image: linear-gradient(to bottom, #2f2f2f, #333333, #000000, #ff5733);
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}}
.stApp::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 255, 0, 0.5); /* Adjust the transparency and color */
    z-index: -1; /* Send it behind other elements */
}}
h1, h2 {{
    color: white;
}}
</style>
'''

# Apply the background image and overlay
st.markdown(overlay_css, unsafe_allow_html=True)

# Load the saved Gradient Boosting model
gb_model = joblib.load('gradient_boosting_classifier.pkl')

# Load the scaler
scaler = joblib.load('standardscaler.pkl')


# Define the Streamlit app

# Center the title using markdown
st.markdown("<h1 style='text-align: center;'>Cardiovascular Disease Prediction App🫀</h1>", unsafe_allow_html=True)

# Sidebar for input fields
st.sidebar.write("""Please fill in the details below to get the prediction from the Gradient Boosting model.""")

# Input fields with emojis
gender = st.sidebar.selectbox("Gender 👤", [1, 2], format_func=lambda x: "Male 👨" if x == 1 else "Female 👩")
age_years = st.sidebar.slider("Age in years 🎂", min_value=1, max_value=120, value=30)
smoke = st.sidebar.selectbox("Do you smoke? 🚬", [0, 1], format_func=lambda x: "No 🚫" if x == 0 else "Yes 👍")
alco = st.sidebar.selectbox("Do you consume alcohol? 🍷", [0, 1], format_func=lambda x: "No 🚫" if x == 0 else "Yes 🍻")
active = st.sidebar.selectbox("Are you physically active? 🏃", [0, 1], format_func=lambda x: "No 🚶" if x == 0 else "Yes 💪")
ap_hi = st.sidebar.slider("Systolic blood pressure (ap_hi) 💖", min_value=50, max_value=200, value=120)
ap_lo = st.sidebar.slider("Diastolic blood pressure (ap_lo) 💔", min_value=30, max_value=120, value=80)
if ap_hi <= ap_lo:
    st.markdown(
        "<p style='color: white;'> ERROR: Systolic blood pressure must be greater than diastolic pressure.</p>",
        unsafe_allow_html=True
    )

cholesterol = st.sidebar.selectbox("Cholesterol level 🧴", [1, 2, 3], 
                                   format_func=lambda x: "Normal 🍏" if x == 1 else "Above Normal 🍔" if x == 2 else "Well Above Normal 🍩")
gluc = st.sidebar.selectbox("Glucose level 🩸", [1, 2, 3], 
                             format_func=lambda x: "Normal 🍏" if x == 1 else "Above Normal 🍔" if x == 2 else "Well Above Normal 🍩")
BMI = st.sidebar.slider("Body Mass Index (BMI) 🏋️", min_value=10.0, max_value=50.0, value=25.0)

# Calculated metrics
pulse_pressure = ap_hi - ap_lo

# Main layout with a long table
st.markdown("<h2 style='text-align: center;'>Your Inputs</h2>", unsafe_allow_html=True)
input_data = {
    "Metric": ["Gender 👤", "Age 🎂", "Do you smoke? 🚬", "Do you consume alcohol? 🍷", "Are you physically active? 🏃", 
               "Systolic Blood Pressure 💖", "Diastolic Blood Pressure 💔", "Cholesterol Level 🧴", "Glucose Level 🩸", 
               "Body Mass Index (BMI) 🏋️", "Pulse Pressure 💥"],
    "Value": ["Male 👨" if gender == 1 else "Female 👩", f"{age_years} years 🎂", "Yes 👍" if smoke == 1 else "No 🚫", 
              "Yes 🍻" if alco == 1 else "No 🚫", "Yes 💪" if active == 1 else "No 🚶",
              ap_hi, ap_lo, "Normal 🍏" if cholesterol == 1 else "Above Normal 🍔" if cholesterol == 2 else "Well Above Normal 🍩",
              "Normal 🍏" if gluc == 1 else "Above Normal 🍔" if gluc == 2 else "Well Above Normal 🍩", 
              BMI, pulse_pressure],
    "Description": ["Biological sex of the individual", 
                    "Age of the individual in years", 
                    "Whether the individual smokes", 
                    "Whether the individual consumes alcohol", 
                    "Whether the individual is physically active", 
                    "The systolic blood pressure (highest pressure when heart beats)", 
                    "The diastolic blood pressure (lowest pressure when heart rests)", 
                    "Cholesterol level in the blood", 
                    "Glucose level in the blood", 
                    "Body Mass Index, a measure of body fat based on height and weight", 
                    "Difference between systolic and diastolic blood pressure"]
}
input_data_df = pd.DataFrame(input_data)

# Display the table without custom styling
st.dataframe(input_data_df)

st.markdown("<h2 style='text-align: center;'>Prediction</h2>", unsafe_allow_html=True)

# One-hot encoding for cholesterol and glucose
chol_1, chol_2, chol_3 = (cholesterol == i for i in range(1, 4))
gluc_1, gluc_2, gluc_3 = (gluc == i for i in range(1, 4))

# Create a numpy array from the input data
input_data = np.array([[gender, ap_hi, ap_lo, smoke, alco, active, age_years, BMI, 
                        chol_1, chol_2, chol_3, gluc_1, gluc_2, gluc_3, 
                        pulse_pressure]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make predictions with the Gradient Boosting model
gb_pred = gb_model.predict(input_data_scaled)
gb_pred_proba = gb_model.predict_proba(input_data_scaled)

# Display the prediction with emojis
if gb_pred[0] == 0:
    st.markdown(f"<p style='color: white; font-size: 18px; text-align: center;'>No Cardiovascular Disease predicted with a probability of {gb_pred_proba[0][0] * 100:.2f}%.</p>", unsafe_allow_html=True)
else:
    st.markdown(f"<p style='color: white; font-size: 18px; text-align: center;'>Cardiovascular Disease predicted with a probability of {gb_pred_proba[0][1] * 100:.2f}%.</p>", unsafe_allow_html=True)

st.sidebar.write("For more information, contact the developer.")
