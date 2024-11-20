import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model and tools
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Streamlit App
st.title("CO2 Emissions Prediction")
st.write("Predict CO2 emissions of a vehicle based on its features.")

# Input Features
model_year = st.number_input("Model Year", min_value=2000, max_value=2025, value=2014, step=1)
vehicle_class = st.selectbox("Vehicle Class", label_encoders["VEHICLECLASS"].classes_)
engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4, step=1)
transmission = st.selectbox("Transmission", label_encoders["TRANSMISSION"].classes_)
fuel_type = st.selectbox("Fuel Type", label_encoders["FUELTYPE"].classes_)
fuel_consumption_city = st.number_input("Fuel Consumption City (L/100 km)", min_value=0.0, max_value=50.0, value=9.9, step=0.1)
fuel_consumption_hwy = st.number_input("Fuel Consumption Hwy (L/100 km)", min_value=0.0, max_value=50.0, value=6.7, step=0.1)
fuel_consumption_comb = st.number_input("Fuel Consumption Comb (L/100 km)", min_value=0.0, max_value=50.0, value=8.5, step=0.1)
fuel_consumption_comb_mpg = st.number_input("Fuel Consumption Comb (mpg)", min_value=0, max_value=100, value=33, step=1)

# Preprocess inputs
input_data = pd.DataFrame({
    "MODELYEAR": [model_year],
    "VEHICLECLASS": [label_encoders["VEHICLECLASS"].transform([vehicle_class])[0]],
    "ENGINESIZE": [engine_size],
    "CYLINDERS": [cylinders],
    "TRANSMISSION": [label_encoders["TRANSMISSION"].transform([transmission])[0]],
    "FUELTYPE": [label_encoders["FUELTYPE"].transform([fuel_type])[0]],
    "FUELCONSUMPTION_CITY": [fuel_consumption_city],
    "FUELCONSUMPTION_HWY": [fuel_consumption_hwy],
    "FUELCONSUMPTION_COMB": [fuel_consumption_comb],
    "FUELCONSUMPTION_COMB_MPG": [fuel_consumption_comb_mpg]
})

# Scale inputs
scaled_input = scaler.transform(input_data)

# Make prediction
if st.button("Predict CO2 Emissions"):
    prediction = model.predict(scaled_input)
    st.success(f"Predicted CO2 Emissions: {prediction[0]:.2f} g/km")