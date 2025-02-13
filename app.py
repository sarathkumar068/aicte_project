import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# ✅ Set Streamlit Page Configuration
st.set_page_config(page_title="Prediction of Disease Outbreaks", layout="wide", page_icon="🧑‍⚕️")

# ✅ Define a relative path for model storage
MODEL_PATH = os.path.join(os.getcwd(), "saved_models")  # Looks for 'saved_models' in the current directory

# ✅ Load saved models with error handling
def load_model(file_name):
    model_path = os.path.join(MODEL_PATH, file_name)
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found: {model_path}")
        return None
    try:
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    except Exception as e:
        st.error(f"⚠️ Error loading model {file_name}: {e}")
        return None

diabetes_model = load_model("diabetes_model.sav")
heart_disease_model = load_model("heart_disease_model.sav")
parkinsons_model = load_model("parkinsons_model.sav")

# ✅ Sidebar menu
with st.sidebar:
    selected = option_menu("Prediction of Disease Outbreaks System",
                           ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
                           menu_icon="hospital-fill",
                           icons=["activity", "heart", "person"],
                           default_index=0)

# ✅ Function to safely convert input to float
def safe_convert(value):
    try:
        return float(value)
    except ValueError:
        return None

# ============================== 🩸 Diabetes Prediction ============================== #
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")

    # Collect user inputs
    col1, col2, col3 = st.columns(3)
    with col1: Pregnancies = st.text_input("Number of Pregnancies")
    with col2: Glucose = st.text_input("Glucose Level")
    with col3: BloodPressure = st.text_input("Blood Pressure Value")
    with col1: SkinThickness = st.text_input("Skin Thickness Value")
    with col2: Insulin = st.text_input("Insulin Level")
    with col3: BMI = st.text_input("BMI Value")
    with col1: DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    with col2: Age = st.text_input("Age of the Person")

    # Predict Diabetes
    if st.button("Diabetes Test Result"):
        if diabetes_model:
            user_input = [safe_convert(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
            if None in user_input:
                st.error("⚠️ Please enter valid numeric values!")
            else:
                diab_prediction = diabetes_model.predict([user_input])
                st.success("✅ The person is diabetic" if diab_prediction[0] == 1 else "✅ The person is not diabetic")
        else:
            st.error("❌ Model not loaded. Check the file path!")

# ============================== ❤️ Heart Disease Prediction ============================== #
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    # Collect user inputs
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

    # Predict Heart Disease
    if st.button("Heart Disease Test Result"):
        if heart_disease_model:
            user_input = [safe_convert(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            if None in user_input:
                st.error("⚠️ Please enter valid numeric values!")
            else:
                heart_prediction = heart_disease_model.predict([user_input])
                st.success("✅ The person has heart disease" if heart_prediction[0] == 1 else "✅ The person does not have heart disease")
        else:
            st.error("❌ Model not loaded. Check the file path!")

# ============================== 🧠 Parkinson's Disease Prediction ============================== #
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    # Collect user inputs
    inputs = []
    labels = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
              "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE",
              "DFA", "Spread1", "Spread2", "D2", "PPE"]
    
    for label in labels:
        inputs.append(st.text_input(label))

    # Predict Parkinson's Disease
    if st.button("Parkinson's Test Result"):
        if parkinsons_model:
            user_input = [safe_convert(x) for x in inputs]
            if None in user_input:
                st.error("⚠️ Please enter valid numeric values!")
            else:
                parkinsons_prediction = parkinsons_model.predict([user_input])
                st.success("✅ The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "✅ The person does not have Parkinson's disease")
        else:
            st.error("❌ Model not loaded. Check the file path!")
