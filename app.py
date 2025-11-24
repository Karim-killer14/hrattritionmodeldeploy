# streamlit_deploy_app.py
"""
Streamlit App for Deployment of HR Attrition Model
Loads model + feature list directly from GitHub.
Allows:
- Upload dataset for batch prediction
- Manual single prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

st.title("🚀 HR Attrition Prediction App")
st.markdown("Model dynamically loads from GitHub. No upload required.")

# ---------------------------
# Load model + features directly from GitHub
# ---------------------------
MODEL_URL = "https://raw.githubusercontent.com/yourname/yourrepo/main/HR_Attrition_Model_20251124_221832.pkl"
FEATURES_URL = "https://raw.githubusercontent.com/yourname/yourrepo/main/HR_Model_Features_20251124_221832.pkl"

st.sidebar.header("Model Load Status")
model = None
features = None

try:
    # Load model
    model_bytes = requests.get(MODEL_URL).content
    model = pickle.load(io.BytesIO(model_bytes))

    # Load features
    feat_bytes = requests.get(FEATURES_URL).content
    features = pickle.load(io.BytesIO(feat_bytes))

    st.sidebar.success("Model & features loaded successfully from GitHub!")
except Exception as e:
    st.sidebar.error(f"Error loading model from GitHub: {e}")

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

        missing_cols = [c for c in features if c not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            df_model = df[features]

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
        input_dict[col] = st.number_input(col, value=0.0)

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
st.info("Model auto-loads from GitHub. Ready for predictions.")
