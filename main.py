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
model_path = r'C:/Users/Sanika/OneDrive/Desktop/Major Project/Diabetes/diabetes_model.sav'
image_path = r'C:/Users/Sanika/OneDrive/Desktop/Major Project/Diabetes/dia.jpg'

# Load the saved model
if os.path.exists(model_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
else:
    st.error("Model file not found. Please check the path.")

# Function for prediction
def diabetes_prediction(input_data):
    # Convert input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

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
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of Patient')

    # Code for prediction
    diagnosis = ''

    # Creating button for prediction
    if st.button('Diabetes Result'):
        try:
            # Validate and convert inputs
            inputs = [
                int(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                int(Age)
            ]
            diagnosis = diabetes_prediction(inputs)
        except ValueError:
            st.error("Please enter valid numeric values.")

    st.success(diagnosis)
    st.write("Diabetes is a chronic condition that occurs when the body cannot properly regulate blood sugar levels, either due to insufficient insulin production (Type 1) or insulin resistance (Type 2). High blood sugar can lead to serious health complications, including heart disease, kidney damage, and nerve issues. Risk factors include obesity, sedentary lifestyle, and a family history of diabetes.")
    
    st.write("### Prevention Points:")
    st.write("""
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
    st.write("These can significantly lower the risk of developing diabetes and its associated complications.")

if __name__ == '__main__':
    main()
