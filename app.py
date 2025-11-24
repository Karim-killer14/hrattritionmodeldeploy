# streamlit_deploy_app.py
"""
Streamlit App for Deployment of HR Attrition Model
Loads:
- Trained model (pkl)
- Feature list (pkl)
Allows:
- Upload dataset or enter single employee features
- Predict attrition probability using saved model
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

st.title("🚀 HR Attrition Prediction App")
st.markdown("Upload the saved model + features to generate attrition predictions.")

# ---------------------------
# Upload model + features
# ---------------------------
st.sidebar.header("Model Files")
uploaded_model = st.sidebar.file_uploader("Upload Model (.pkl)", type=["pkl"])
uploaded_features = st.sidebar.file_uploader("Upload Features (.pkl)", type=["pkl"])

# ---------------------------
# Load model and feature list
# ---------------------------
model = None
features = None

if uploaded_model and uploaded_features:
    try:
        model = pickle.load(uploaded_model)
        features = pickle.load(uploaded_features)
        st.sidebar.success("Model & features loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading files: {e}")

# ---------------------------
# Prediction Modes
# ---------------------------
mode = st.radio("Select Prediction Mode", ["Upload CSV for Batch Prediction", "Manual Single Prediction"])

# ---------------------------
# Batch Prediction Mode
# ---------------------------
if model is not None and features is not None and mode == "Upload CSV for Batch Prediction":
    uploaded_data = st.file_uploader("Upload Employee Data CSV", type=["csv"])

    if uploaded_data:
        df = pd.read_csv(uploaded_data)
        st.write("### Preview")
        st.dataframe(df.head())

        # Ensure expected columns
        missing_cols = [c for c in features if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in features]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            # Keep only needed columns
            df_model = df[features]

            # Predict
            if hasattr(model, "predict_proba"):
                df["Attrition_Probability"] = model.predict_proba(df_model)[:, 1]
            else:
                df["Attrition_Probability"] = model.predict(df_model)

            st.success("Predictions generated!")
            st.dataframe(df)

            st.download_button(
                "📥 Download Predictions",
                data=df.to_csv(index=False),
                file_name="attrition_predictions.csv"
            )

# ---------------------------
# Manual Single Prediction Mode
# ---------------------------
if model is not None and features is not None and mode == "Manual Single Prediction":
    st.write("### Enter Employee Attributes")

    input_dict = {}

    for col in features:
        # Basic numeric input fallback
        input_dict[col] = st.number_input(col, value=0.0)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    if st.button("Predict Attrition"):
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[:, 1][0]
            else:
                prob = float(model.predict(input_df)[0])

            st.metric("Attrition Probability", f"{prob:.2%}")

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ---------------------------
# Footer
# ---------------------------
st.info("Upload your trained model + feature list to begin predictions.")
