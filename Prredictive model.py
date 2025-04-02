# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:20:19 2025

@author: user
"""

import numpy as np
import pickle

#3. Loading the Trained Logistic Regression Model
loaded_model = pickle.load(open("C:/Users/user/Python Scripts/trained_model.sav",'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

