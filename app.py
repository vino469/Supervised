import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression

# ---------------------
# Function to load or train model
# ---------------------
def load_or_train_model(X_train=None, y_train=None):
    model_path = "health_risk_model.pkl"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("⚠️ Training a new model...")
        
        # If no data provided, use dummy dataset
        if X_train is None or y_train is None:
            X_train = np.array([
                [25, 22.5, 120, 200],
                [30, 28.0, 130, 210],
                [45, 30.0, 140, 220],
                [50, 27.5, 150, 230]
            ])
            y_train = np.array([0, 1, 1, 0])
        
        model = LinearRegression()
        model.fit(X_train, y_train)
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

numeric_cols = []
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("Preview of uploaded dataset:")
    st.dataframe(df.head())

    # Select only numeric columns for prediction
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.warning("❌ No numeric columns detected for prediction. Using dummy data.")
        numeric_cols = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
else:
    st.info("Please upload a CSV file to get started.")
    # Dummy features
    numeric_cols = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]

# ---------------------
# Prepare training data (if CSV has at least one numeric column)
# ---------------------
if df is not None and len(numeric_cols) > 0:
    X_train = df[numeric_cols].values
    # Create dummy target if CSV has no 'target' column
    y_train = np.zeros(X_train.shape[0])
else:
    X_train = None
    y_train = None

# Load or train model
model = load_or_train_model(X_train, y_train)

# ---------------------
# Dynamic numeric input fields
# ---------------------
st.write("Enter values for prediction:")
inputs = []
for feature in numeric_cols:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

# ---------------------
# Predict button
# ---------------------
if st.button("Predict Health Risk"):
    X_input = np.array([inputs])
    try:
        prediction = model.predict(X_input)[0]
        st.success(f"Predicted Health Risk: {prediction:.2f}")
    except ValueError as e:
        st.error(f"Error: {e}")
        st.info("Check that the number of input features matches the model.")
