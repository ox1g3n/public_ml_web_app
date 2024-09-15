# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:51:39 2024

@author: Aagnik
"""

import pickle
import streamlit as st
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")



# Get the absolute path of the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the models and scalers using absolute paths
diabetes_model = pickle.load(open(os.path.join(script_dir, 'DiabetesPrediction.sav'), 'rb'))
diabetes_scaler = pickle.load(open(os.path.join(script_dir, 'diabetesscaler.pkl'), 'rb'))

heart_model = pickle.load(open(os.path.join(script_dir, 'HeartDisease.sav'), 'rb'))

parkinsons_model = pickle.load(open(os.path.join(script_dir, 'ParkinsonsPrediction.sav'), 'rb'))
parkinsons_scaler = pickle.load(open(os.path.join(script_dir, 'scaler.pkl'), 'rb'))



# Helper functions
def display_selected(text):
    st.markdown(f'<p style="font-size:24px;">{text}</p>', unsafe_allow_html=True)

def get_option_with_icon(icon, label):
    return f'{icon} {label}'

# List of options with corresponding icons
options = [
    get_option_with_icon('🩸', 'Diabetes prediction'),
    get_option_with_icon('❤️', 'Heart Disease prediction'),
    get_option_with_icon('🧠', 'Parkinsons Disease prediction')
]

# Sidebar selectbox with icons
selected_option = st.sidebar.selectbox(
    'Multiple Disease Prediction System',
    options
)

# Handling the selected option
if 'Diabetes' in selected_option:
    st.title('Diabetes prediction selected.')
    st.write('Please take a standardised blood test before using this app')
    st.write('This page is for women only')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level(mg/dL)[0-200]')

    with col3:
        BloodPressure = st.text_input('Systolic Blood Pressure value (mmHg)[80-120]')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value(mm)[0-60]')

    with col2:
        Insulin = st.text_input('Insulin Level(µIU/mL)[0-1000]')

    with col3:
        BMI = st.text_input('BMI value[15-50]')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value[0-1]')

    with col2:
        Age = st.text_input('Age of the Person')

    # Prediction for Diabetes
    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        user_input = np.asarray(user_input).reshape(1, -1)
        user_input = diabetes_scaler.transform(user_input)
        diab_prediction = diabetes_model.predict(user_input)

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

elif 'Heart' in selected_option:
    st.title('Heart Disease prediction selected.')
    st.write('Standardised test required')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex[0-Male,1-Female]')

    with col3:
        cp = st.text_input('Chest Pain types(Under pressure)[0,1,2,3]')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure Systolic(mmHg)[80-120]')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl[100-400]')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl[0-yes,1-no]')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results[0 or 1]')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved(bpm)[100-200]')

    with col3:
        exang = st.text_input('Exercise Induced Angina[0-no,1-yes]')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise(mm)[0-3]')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment[0,1,2]')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy[0,1,2,3]')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')

    # Prediction for Heart Disease
    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        heart_prediction = heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

elif 'Parkinsons' in selected_option:
    st.title('Parkinsons Disease prediction selected.')
    st.write('Must be used by medical professionals only')
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    # Prediction for Parkinson's
    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                      APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        user_input = np.asarray(user_input).reshape(1, -1)
        user_input = parkinsons_scaler.transform(user_input)
        parkinsons_prediction = parkinsons_model.predict(user_input)

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = 'The person has parkinsons'
        else:
            parkinsons_diagnosis = 'The person does not have parkinsons'

    st.success(parkinsons_diagnosis)
