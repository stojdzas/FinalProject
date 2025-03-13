import streamlit as st
import joblib
import pandas as pd
import numpy as np
from fpdf import FPDF
import os
from PIL import Image

# Load the saved model
model = joblib.load('ad_predictor.pkl')

# List of Alzheimer’s facts
facts = [
    "🧠 **Alzheimer's disease affects over 55 million people worldwide!**",
    "🧠 **Alzheimer’s is a progressive disease, meaning symptoms worsen over time.**",
    "🧠 **Lifestyle changes, such as a healthy diet, exercise, and mental stimulation, may help reduce the risk of Alzheimer’s.**"
]

# Initialize session state for tracking clicks
if "click_count" not in st.session_state:
    st.session_state.click_count = -1

# Display the image
st.image("brain_image.jpg", use_column_width=True)

# Button that simulates image click
if st.button("Did You Know?"):
    if st.session_state.click_count < len(facts) - 1:
        st.session_state.click_count += 1  # Show the next fact
    else:
        st.session_state.click_count = -1  # Reset back to only showing the button

# Show facts based on the number of clicks
if st.session_state.click_count >= 0:  
    st.write(facts[st.session_state.click_count])  # Display only the current fact

# App Title
st.title("Alzheimer's Disease Predictor")

# Description
st.write("""
    This app predicts whether a person has Alzheimer's disease. Using a machine learning model trained on clinical and lifestyle factors, 
    the app assists in detecting Alzheimer’s disease and supporting decision-making.
    
    Enter the data below to get a prediction.
""")

# Category 1: Demographic Information
with st.expander("Demographic Information"):
    # Age input
    feature1 = st.slider("Age:", 0, 100)
    # Gender input
    feature2 = st.selectbox("Gender:", ["Female", "Male"])
    # Ethnicity input
    feature3 = st.selectbox("Ethnicity:", ["Caucasian", "African American", "Asian", "Other"])
    # Education level input
    feature4 = st.selectbox("Education Level:", ["None", "High School", "Bachelor's", "Higher"])

# Category 2: Lifestyle Factors
with st.expander("Lifestyle Factors"):
    feature5 = st.slider("BMI:", 15, 40)
    feature6 = st.selectbox("Smoking:", ["No", "Yes"])
    feature7 = st.slider("Alcohol Consumption:", 0, 20)
    feature8 = st.slider("Physical Activity:", 0, 10)
    feature9 = st.slider("Diet Quality:", 0, 10)
    feature10 = st.slider("Sleep Quality:", 4, 10)

# Category 3: Medical History
with st.expander("Medical History"):
    feature11 = st.selectbox("Family history of Alzheimer's Disease:", ["No", "Yes"])
    feature12 = st.selectbox("Cardiovascular Disease:", ["No", "Yes"])
    feature13 = st.selectbox("Diabetes:", ["No", "Yes"])
    feature14 = st.selectbox("Depression:", ["No", "Yes"])
    feature15 = st.selectbox("Head Injury:", ["No", "Yes"])
    feature16 = st.selectbox("Hypertension:", ["No", "Yes"])

# Category 4: Clinical Measurements
with st.expander("Clinical Measurements"):
    feature17 = st.slider("Systolic Blood Pressure:", 90, 180)
    feature18 = st.slider("Diastolic Blood Pressure:", 60, 120)
    feature19 = st.slider("Cholesterol Total:", 150, 300)
    feature20 = st.slider("Cholesterol LDL:", 50, 200)
    feature21 = st.slider("Cholesterol HDL:", 20, 100)
    feature22 = st.slider("Cholesterol Triglycerides:", 50, 400)

# Category 5: Cognitive and Functional Assessments
with st.expander("Cognitive and Functional Assessments"):
    feature23 = st.slider("MMSE (Mini-Mental State Examination):", 0, 30)
    feature24 = st.slider("Functional Assessment Score:", 0, 10)
    feature25 = st.selectbox("Memory Complaints:", ["No", "Yes"])
    feature26 = st.selectbox("Behavioral Problems:", ["No", "Yes"])
    feature27 = st.slider("ADL (Activities of Daily Living) Score:", 0, 10)

# Symptoms
with st.expander("Symptoms"):
    feature28 = st.selectbox("Confusion:", ["No", "Yes"])
    feature29 = st.selectbox("Disorientation:", ["No", "Yes"])
    feature30 = st.selectbox("Personality Changes:", ["No", "Yes"])
    feature31 = st.selectbox("Difficulty Completing Tasks:", ["No", "Yes"])
    feature32 = st.selectbox("Forgetfulness:", ["No", "Yes"])

# Encode categorical features as numbers
gender_map = {"Female": 1, "Male": 0}
ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
education_map = {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}
binary_map = {"No": 0, "Yes": 1}  # For Yes/No features

# Convert categorical inputs to numerical values
feature2 = gender_map[feature2]  # Gender
feature3 = ethnicity_map[feature3]  # Ethnicity
feature4 = education_map[feature4]  # Education Level
feature6 = binary_map[feature6]  # Smoking
feature11 = binary_map[feature11]  # Family History
feature12 = binary_map[feature12]  # Cardiovascular Disease
feature13 = binary_map[feature13]  # Diabetes
feature14 = binary_map[feature14]  # Depression
feature15 = binary_map[feature15]  # Head Injury
feature16 = binary_map[feature16]  # Hypertension
feature25 = binary_map[feature25]  # Memory Complaints
feature26 = binary_map[feature26]  # Behavioral Problems
feature28 = binary_map[feature28]  # Confusion
feature29 = binary_map[feature29]  # Disorientation
feature30 = binary_map[feature30]  # Personality Changes
feature31 = binary_map[feature31]  # Difficulty Completing Tasks
feature32 = binary_map[feature32]  # Forgetfulness

# Combine all input data into a DataFrame
data = pd.DataFrame([[
    feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10,
    feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20,
    feature21, feature22, feature23, feature24, feature25, feature26, feature27, feature28, feature29, feature30,
    feature31, feature32
]], columns=[
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity",
    "DietQuality", "SleepQuality", "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression", 
    "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", 
    "CholesterolHDL", "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "MemoryComplaints", 
    "BehavioralProblems", "ADL", "Confusion", "Disorientation", "PersonalityChanges", 
    "DifficultyCompletingTasks", "Forgetfulness"
])

# Prediction button
if st.button('Diagnose'):
    prediction = model.predict(data)
    if prediction == 0:
        prediction_result = "Healthy"
    else:
        prediction_result = "Alzheimer's Disease"
        
    st.write(f"Diagnose: **{prediction_result}**")
    
    def generate_pdf(prediction, data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Alzheimer's Prediction Report", ln=True, align='C')
        pdf.cell(200, 10, f"Prediction: {prediction}", ln=True, align='L')
        
        for col, val in data.iloc[0].items():
            pdf.cell(200, 10, f"{col}: {val}", ln=True, align='L')

        pdf_output_path = "report.pdf"
        pdf.output(pdf_output_path)

        return pdf_output_path

    pdf_file = generate_pdf(prediction_result, data)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="Download Report",
            data=f,
            file_name="alzheimers_report.pdf",
            mime="application/pdf"
        )
    
    os.remove(pdf_file)

if st.button('Feature Importance Insights'):
    # Open the image using PIL
    image1 = Image.open('New_Feature_Selection.png')  
    image2 = Image.open('MMSE.png')
    image3 = Image.open('FA.png')
    image4 = Image.open('ADL.png')
    # Display the image
    st.image(image1, caption="", use_column_width=True)
    st.image(image2, caption="", use_column_width=True)
    st.image(image3, caption="", use_column_width=True)
    st.image(image4, caption="", use_column_width=True)

# Optional: Show the input data to the user
st.write("Input Data:", data)


    
    
