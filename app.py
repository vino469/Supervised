import streamlit as st
import pandas as pd
import numpy as np
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
        st.warning("⚠️ Model not found. Training a new model...")
        
        # Example dataset (replace with your real dataset)
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([10, 20, 30, 40])
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, model_path)
        st.success("✅ New model trained and saved!")
    
    return model

# ---------------------
# Streamlit App
# ---------------------
st.set_page_config(page_title="Patient Health Risk Predictor", layout="centered")
st.title("🩺 Patient Health Risk Predictor")

st.write("Enter patient details to predict health risk:")

# Load or train model
model = load_or_train_model()

# Example input fields (adjust according to your dataset)
feature1 = st.number_input("Feature 1", value=1.0)
feature2 = st.number_input("Feature 2", value=2.0)

# Predict button
if st.button("Predict Health Risk"):
    X_input = np.array([[feature1, feature2]])
    prediction = model.predict(X_input)[0]
    st.success(f"Predicted Health Risk: {prediction:.2f}")
