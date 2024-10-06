# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:23:31 2024

@author: Sanika
"""

import numpy as np
import pickle
import streamlit as st
from PIL import Image 

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Sanika/OneDrive/Desktop/Major Project\Diabetes/diabetes_model.sav', 'rb'))

#function creation for Prediction
def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    st.title('DIABETES PREDICTION SYSTEM')
    col1, col2 = st.columns([1,5])  # Adjust column sizes as needed

# Load and resize the image
    with col2:  # Place the image in the second column (right)
        try:
            image = Image.open('C:/Users/Sanika/OneDrive/Desktop/Major Project/Diabetes/dia.jpg')
            max_size = (400, 350)  # Max size you want
            image.thumbnail(max_size)  # Maintain aspect ratio
            st.image(image, use_column_width=False)  # Display the image centered
        except Exception as e:
            st.error(f"Error loading image: {e}")		
    
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure')
    SkinThickness=st.text_input('Skin Thickness value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function value')
    Age=st.text_input('Age of Patient')

#code for predection
    diagonsis=''	
    
    #creating button for prediction
    if st.button('Diabetes Result'):
        diagonsis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagonsis)
    st.write("Diabetes is a chronic condition that occurs when the body cannot properly regulate blood sugar levels, either due to insufficient insulin production (Type 1) or insulin resistance (Type 2). High blood sugar can lead to serious health complications, including heart disease, kidney damage, and nerve issues. Risk factors include obesity, sedentary lifestyle, and a family history of diabetes.")
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
    st.write("These can significantly lower the risk of developing diabetes and its associated complications.")
  
    
if __name__ == '__main__':
    main()