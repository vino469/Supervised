import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# ---------------------
# Page config
# ---------------------
st.set_page_config(page_title="Patient Health Risk Predictor", layout="centered")
st.title("🩺 Patient Health Risk Predictor")
st.write("Upload your CSV dataset below (drag & drop or browse):")

# ---------------------
# File uploader
# ---------------------
uploaded_file = st.file_uploader(
    "Drag and drop a CSV file here or click to browse",
    type=["csv"]
)

# Initialize variables
numeric_cols = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]  # default features
model = LinearRegression()
trained_cols = numeric_cols.copy()

# ---------------------
# Load CSV & train model if possible
# ---------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("Preview of uploaded dataset:")
    st.dataframe(df.head())

    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) == 0:
        st.warning("❌ No numeric columns detected. Using default dummy features.")
        numeric_cols = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
    else:
        trained_cols = numeric_cols.copy()

        # Check for 'target' column
        if "target" in df.columns:
            X_train = df[numeric_cols].values
            y_train = df["target"].values
            model.fit(X_train, y_train)
            joblib.dump(model, "health_risk_model.pkl")
            st.success("✅ Model trained successfully using uploaded CSV!")
        else:
            st.warning(" CSV has no 'target' column. Using dummy model instead.")

# ---------------------
# Dummy model if no CSV or no target
# ---------------------
if not uploaded_file or "target" not in df.columns:
    X_dummy = np.array([
        [25, 22.5, 120, 200],
        [30, 28.0, 130, 210],
        [45, 30.0, 140, 220],
        [50, 27.5, 150, 230]
    ])
    y_dummy = np.array([0, 1, 1, 0])
    model.fit(X_dummy, y_dummy)
    joblib.dump(model, "health_risk_model.pkl")
    st.info("ℹ️ Using dummy model with default features.")

# ---------------------
# Dynamic numeric input fields
# ---------------------
st.write("Enter values for prediction:")
inputs = []
for feature in trained_cols:
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
