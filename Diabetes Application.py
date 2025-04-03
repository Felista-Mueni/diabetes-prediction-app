import numpy as np
import pickle
import streamlit as st

# 3. Loading the Trained Logistic Regression Model
loaded_model = pickle.load(open("C:/Users/user/Python Scripts/trained_model.sav", 'rb'))


# Create a function to for prediction

def diabetic_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    # Giving a title to UI Interface
    st.title('Diabetic Prediction Web App')

    # Getting the input data for the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thinckness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age')

    # Code for prediction
    diagnosis = ''

    # Creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetic_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
