# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:09:48 2025

@author: pikuu
"""
import pickle
with open("scaler.pkl", "rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open("ridge.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

import pandas as pd

# Assuming you have the original feature names from X_train
feature_names = X_train.columns  

# Convert new input into a DataFrame with column names
X_new = pd.DataFrame([[30, 40, 5, 0.1, 90, 100, 200, 5, 1]], columns=feature_names)

# Transform using the loaded scaler
X_new_scaled = loaded_scaler.transform(X_new)

# Make prediction
prediction = loaded_model.predict(X_new_scaled)
print("Predicted Value:", prediction[0])

