import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np  # Added for numeric conversions

# Set Streamlit page configuration
st.set_page_config(page_title="Prediction of Disease Outbreaks", layout="wide", page_icon="🧑‍⚕️")

# Define paths to saved models (Use raw strings for Windows compatibility)
MODEL_PATH = r"C:\Users\ELCOT\Desktop\aicte_project\saved_models"

# Load saved models with error handling
def load_model(file_name):
    try:
        with open(os.path.join(MODEL_PATH, file_name), "rb") as model_file:
            return pickle.load(model_file)
    except Exception as e:
        st.error(f"Error loading model {file_name}: {e}")
        return None

diabetes_model = load_model("diabetes_model.sav")
heart_disease_model = load_model("heart_disease_model.sav")
parkinsons_model = load_model("parkinsons_model.sav")

# Sidebar menu
with st.sidebar:
    selected = option_menu("Prediction of Disease Outbreaks System",
                           ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
                           menu_icon="hospital-fill",
                           icons=["activity", "heart", "person"],
                           default_index=0)

# Function to convert user input safely
def safe_convert(value, dtype=float):
    try:
        return dtype(value)
    except ValueError:
        return None  # Return None for invalid input

# 🩸 Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")

    # User input
    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input("Number of Pregnancies")
    with col2: Glucose = st.text_input("Glucose Level")
    with col3: BloodPressure = st.text_input("Blood Pressure Value")
    with col1: SkinThickness = st.text_input("Skin Thickness Value")
    with col2: Insulin = st.text_input("Insulin Level")
    with col3: BMI = st.text_input("BMI Value")
    with col1: DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    with col2: Age = st.text_input("Age of the Person")

    # Prediction
    if st.button("Diabetes Test Result"):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [safe_convert(x) for x in user_input]

        if None in user_input:
            st.error("Please enter valid numeric values!")
        else:
            diab_prediction = diabetes_model.predict([user_input])
            st.success("The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic")

# ❤️ Heart Disease Prediction
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    # User input
    col1, col2, col3 = st.columns(3)
    with col1: age = st.text_input("Age")
    with col2: sex = st.text_input("Sex (1=Male, 0=Female)")
    with col3: cp = st.text_input("Chest Pain Type")
    with col1: trestbps = st.text_input("Resting Blood Pressure")
    with col2: chol = st.text_input("Serum Cholesterol in mg/dl")
    with col3: fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)")
    with col1: restecg = st.text_input("Resting Electrocardiographic Results")
    with col2: thalach = st.text_input("Maximum Heart Rate Achieved")
    with col3: exang = st.text_input("Exercise Induced Angina (1=Yes, 0=No)")
    with col1: oldpeak = st.text_input("ST Depression Induced by Exercise")
    with col2: slope = st.text_input("Slope of the Peak Exercise ST Segment")
    with col3: ca = st.text_input("Major Vessels Colored by Fluoroscopy")
    with col1: thal = st.text_input("Thal (0=Normal, 1=Fixed Defect, 2=Reversible Defect)")

    # Prediction
    if st.button("Heart Disease Test Result"):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [safe_convert(x) for x in user_input]

        if None in user_input:
            st.error("Please enter valid numeric values!")
        else:
            heart_prediction = heart_disease_model.predict([user_input])
            st.success("The person has heart disease" if heart_prediction[0] == 1 else "The person does not have heart disease")

# 🧠 Parkinson's Disease Prediction
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    # User input
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: fo = st.text_input("MDVP:Fo(Hz)")
    with col2: fhi = st.text_input("MDVP:Fhi(Hz)")
    with col3: flo = st.text_input("MDVP:Flo(Hz)")
    with col4: Jitter_percent = st.text_input("MDVP:Jitter(%)")
    with col5: Jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
    with col1: RAP = st.text_input("MDVP:RAP")
    with col2: PPQ = st.text_input("MDVP:PPQ")
    with col3: DDP = st.text_input("Jitter:DDP")
    with col4: Shimmer = st.text_input("MDVP:Shimmer")
    with col5: Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
    with col1: APQ3 = st.text_input("Shimmer:APQ3")
    with col2: APQ5 = st.text_input("Shimmer:APQ5")
    with col3: APQ = st.text_input("MDVP:APQ")
    with col4: DDA = st.text_input("Shimmer:DDA")
    with col5: NHR = st.text_input("NHR")
    with col1: HNR = st.text_input("HNR")
    with col2: RPDE = st.text_input("RPDE")
    with col3: DFA = st.text_input("DFA")
    with col4: spread1 = st.text_input("Spread1")
    with col5: spread2 = st.text_input("Spread2")
    with col1: D2 = st.text_input("D2")
    with col2: PPE = st.text_input("PPE")

    # Prediction
    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        user_input = [safe_convert(x) for x in user_input]

        if None in user_input:
            st.error("Please enter valid numeric values!")
        else:
            parkinsons_prediction = parkinsons_model.predict([user_input])
            st.success("The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease")
