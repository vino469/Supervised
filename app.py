import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Patient Health Risk Predictor", layout="centered")
st.title("🩺 Patient Health Risk Predictor")
st.write("Upload your CSV dataset below (drag & drop or browse):")

uploaded_file = st.file_uploader("Drag and drop a CSV file here or click to browse", type=["csv"])

dummy_features = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
trained_cols = dummy_features.copy()
model = LinearRegression()
use_dummy = True  # fallback

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) >= 2:
        # If 'target' exists, use it; otherwise use last numeric column
        if "target" in df.columns:
            target_col = "target"
        else:
            target_col = numeric_cols[-1]  # last numeric column as target
            st.info(f"ℹ️ No 'target' column found. Using '{target_col}' as target.")

        feature_cols = [c for c in numeric_cols if c != target_col]

        X_train = df[feature_cols].values
        y_train = df[target_col].values

        model.fit(X_train, y_train)
        trained_cols = feature_cols.copy()  # inputs match features
        joblib.dump(model, "health_risk_model.pkl")
        st.success("✅ Model trained successfully from CSV!")
        use_dummy = False
    else:
        st.warning(" Not enough numeric columns. Using dummy model instead.")

# ---------------------
# Dummy model fallback
# ---------------------
if use_dummy:
    X_dummy = np.array([
        [25, 22.5, 120, 200],
        [30, 28.0, 130, 210],
        [45, 30.0, 140, 220],
        [50, 27.5, 150, 230]
    ])
    y_dummy = np.array([0, 1, 1, 0])
    model.fit(X_dummy, y_dummy)
    trained_cols = dummy_features.copy()
    joblib.dump(model, "health_risk_model.pkl")
    st.info("ℹ️ Using dummy model with 4 features.")

# ---------------------
# Input fields
# ---------------------
st.write("Enter values for prediction:")
inputs = []
for feature in trained_cols:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

# ---------------------
# Prediction
# ---------------------
if st.button("Predict Health Risk"):
    X_input = np.array([inputs])
    prediction = model.predict(X_input)[0]
    st.success(f"Predicted Health Risk: {prediction:.2f}")
