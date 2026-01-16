import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression

# ---------------------
# Function to load or train model
# ---------------------
def load_or_train_model(df=None):
    model_path = "health_risk_model.pkl"
    
    if os.path.exists(model_path) and df is None:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("⚠️ Training a new model...")

        if df is None:
            # Dummy numeric data if no CSV uploaded
            X_train = np.array([
                [25, 22.5, 120, 200],
                [30, 28.0, 130, 210],
                [45, 30.0, 140, 220],
                [50, 27.5, 150, 230]
            ])
            y_train = np.array([0, 1, 1, 0])
        else:
            if 'target' not in df.columns:
                st.error("❌ Your CSV must have a column named 'target'")
                st.stop()
            # Only numeric features
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            numeric_cols.remove('target')
            X_train = df[numeric_cols].values
            y_train = df['target'].values

        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, model_path)
        st.success("✅ Model trained and saved!")

    return model

# ---------------------
# Streamlit App
# ---------------------
st.set_page_config(page_title="Patient Health Risk Predictor", layout="centered")
st.title("🩺 Patient Health Risk Predictor")

st.write("Upload your CSV dataset below (drag & drop or browse):")

# ---------------------
# Drag & drop file uploader
# ---------------------
uploaded_file = st.file_uploader(
    label="Drag and drop a CSV file here or click to browse",
    type=["csv"]
)

df = None
numeric_cols = []

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("Preview of uploaded dataset:")
    st.dataframe(df.head())
    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
else:
    st.info("Please upload a CSV file to get started.")

# Load or train the model
model = load_or_train_model(df)

# ---------------------
# Dynamic numeric input fields
# ---------------------
if df is not None and numeric_cols:
    st.write("Enter values for prediction:")
    inputs = []
    for feature in numeric_cols:
        value = st.number_input(f"{feature}", value=0.0)
        inputs.append(value)
    
    # Predict button
    if st.button("Predict Health Risk"):
        X_input = np.array([inputs])
        try:
            prediction = model.predict(X_input)[0]
            st.success(f"Predicted Health Risk: {prediction:.2f}")
        except ValueError as e:
            st.error(f"Error: {e}")
            st.info("Check that the number of input features matches the model.")
elif df is not None and not numeric_cols:
    st.warning("No numeric columns detected for prediction.")
