import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.linear_model import LinearRegression

# ---------------------
# Function to load or train model
# ---------------------
def load_or_train_model():
    model_path = "health_risk_model.pkl"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("⚠️ Model not found. Training a new dummy model...")
        
        # Dummy dataset (replace with your real dataset)
        # Example: 4 features
        X_train = np.array([
            [25, 22.5, 120, 200],
            [30, 28.0, 130, 210],
            [45, 30.0, 140, 220],
            [50, 27.5, 150, 230]
        ])
        y_train = np.array([0, 1, 1, 0])  # Example target: 0=low risk, 1=high risk
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(model, model_path)
        st.success("✅ Dummy model trained and saved as health_risk_model.pkl!")
    
    return model

# ---------------------
# Streamlit App
# ---------------------
st.set_page_config(page_title="Patient Health Risk Predictor", layout="centered")
st.title("🩺 Patient Health Risk Predictor")

# Load or train the model
model = load_or_train_model()

# Display expected features
st.write(f"Model expects {model.n_features_in_} features.")

# ---------------------
# Create dynamic input fields based on model features
# ---------------------
# If you know your feature names, you can replace these with actual names
feature_names = [f"Feature {i+1}" for i in range(model.n_features_in_)]  

inputs = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

# ---------------------
# Predict button
# ---------------------
if st.button("Predict Health Risk"):
    X_input = np.array([inputs])  # Ensure shape is (1, n_features)
    try:
        prediction = model.predict(X_input)[0]
        st.success(f"Predicted Health Risk: {prediction:.2f}")
    except ValueError as e:
        st.error(f"Error: {e}")
        st.info("Check that the number of input features matches the model.")
