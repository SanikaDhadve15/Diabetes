# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:23:31 2024

@author: Sanika
"""

import numpy as np
import pickle
import streamlit as st
from PIL import Image 
import os

# Define paths
model_path ='diabetes_model.sav'
image_path = 'dia.jpg'

# Load the saved model
loaded_model = None

if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    st.success("Model loaded successfully.")
else:
    st.error("Model file not found. Please check the path.")

# Function for prediction
def diabetes_prediction(input_data):
    if loaded_model is None:
        return "Model not loaded."

    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data_as_numpy_array)
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'

def main():
    st.title('Diabetes Prediction System')
    col1, col2 = st.columns([1, 5])

    # Load and resize the image
    with col2:
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                max_size = (400, 350)
                image.thumbnail(max_size)
                st.image(image, use_column_width=False)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        else:
            st.error("Image file not found. Please check the path.")

    # Input fields
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    Glucose = st.number_input('Glucose Level', min_value=0.0, step=0.1)
    BloodPressure = st.number_input('Blood Pressure', min_value=0.0, step=0.1)
    SkinThickness = st.number_input('Skin Thickness Value', min_value=0.0, step=0.1)
    Insulin = st.number_input('Insulin Level', min_value=0.0, step=0.1)
    BMI = st.number_input('BMI Value', min_value=0.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function Value', min_value=0.0, step=0.01)
    Age = st.number_input('Age of Patient', min_value=0, step=1)

    # Code for prediction
    diagnosis = ''

    # Creating button for prediction
    if st.button('Diabetes Result'):
        inputs = [
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]
        diagnosis = diabetes_prediction(inputs)

    if diagnosis:
        st.success(diagnosis)

    st.write("Diabetes is a chronic condition that occurs when the body cannot properly regulate blood sugar levels...")
    
    st.write("### Prevention Points:")
    st.write("""\
        1. Healthy Diet
        2. Regular Physical Activity
        3. Weight Management
        4. Regular Health Screenings    
        5. Limit Sugary Beverages  
        6. Stay Hydrated
        7. Avoid Smoking 
        8. Manage Stress
        9. Get Enough Sleep
    """)

if __name__ == '__main__':
    main()
