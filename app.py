# streamlit_deploy_app.py (UPDATED WITH FULL PREPROCESSING)
"""
Streamlit App for Deployment of HR Attrition Model
Loads model + feature list directly from GitHub.
Accepts simple HR CSV and transforms it into the full engineered feature set.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io
import re

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

st.title("🚀 HR Attrition Prediction App")
st.markdown("Model dynamically loads from GitHub. Supports simple HR CSV input.")

# ---------------------------
# URLs for model + features from GitHub
# ---------------------------
MODEL_URL = "https://raw.githubusercontent.com/Karim-killer14/hrattritionmodeldeploy/main/HR_Attrition_Model_20251124_221832.pkl"
FEATURES_URL = "https://raw.githubusercontent.com/Karim-killer14/hrattritionmodeldeploy/main/HR_Model_Features_20251124_221832.pkl"

st.sidebar.header("Model Load Status")
model = None
features = None

try:
    # Load model from GitHub
    model_bytes = requests.get(MODEL_URL).content
    model = pickle.load(io.BytesIO(model_bytes))

    # Load feature list from GitHub
    feat_bytes = requests.get(FEATURES_URL).content
    features = pickle.load(io.BytesIO(feat_bytes))

    st.sidebar.success("Model & features loaded successfully from GitHub!")
except Exception as e:
    st.sidebar.error(f"Error loading model from GitHub: {e}")

# --------------------------------------------------------
# SANITIZER — MATCHES TRAINING PIPELINE
# --------------------------------------------------------
def sanitize_columns_df(df):
    df = df.copy()
    df.columns = df.columns.str.replace(r'[\[\]<>() ]', '_', regex=True)
    return df

def safe_num(x):
    return pd.to_numeric(x, errors="coerce")

# --------------------------------------------------------
# FULL PREPROCESSING PIPELINE — SIMPLE CSV → MODEL FEATURES
# --------------------------------------------------------
def preprocess_for_model(df_raw, features_list):

    df = sanitize_columns_df(df_raw)

    # Convert numeric columns where possible
    for col in df.columns:
        df[col] = safe_num(df[col]).fillna(df[col])

    # Output dataframe with required features
    out = pd.DataFrame(index=df.index)

    # RAW SIMPLE COLUMNS YOU PROVIDED
    raw_direct_cols = [
        'Employee_ID','National_ID','Insurance_Number','Mobile','Email',
        'Department','Job_Title','Working_Conditions','Starting_Date','Duration',
        'Basic_Salary','Other_Allowances','Gross_Salary','No_of_Working_Days',
        'Value_of_Working_Day','Value_of_Working_Hour','Day_Overtime_Hours',
        'Day_Overtime_Value','Night_Overtime_Hours','Night_Overtime_Value',
        'Total_Overtime_Calc','Holiday_Hours','Holiday_Hours_Value',
        'Additional_Holidays__Formal_or_Weekly_2_','Annual_Personal_Exemption',
        'Mobile_Allowance','Bonus_Incentive','Other_Additions','Total_Addition_Net',
        'Other_Deductions_Amount','Penalty_Days','Penalty_Value','Unpaid_Leave_Days',
        'Unpaid_Leave_Value','Tardiness_Hours','Tardiness_Value','Absence_Days',
        'Absence_Value','Employee_SI_Share_11_','Employer_SI_Share_18_75_',
        'Due_Income_Tax','Total_Deductions','Net_Salary_Final'
    ]

    # Fill direct matches
    for col in raw_direct_cols:
        sanitized = col
        if sanitized in df.columns:
            out[sanitized] = safe_num(df[sanitized]).fillna(0)
        else:
            out[sanitized] = 0

    # Derived features (best-effort formulas)
    # -------------------------------------------------------

    # Total overtime hours
    if 'Total_Overtime_Hours' in features_list:
        if 'Total_Overtime_Hours' in df.columns:
            out['Total_Overtime_Hours'] = safe_num(df['Total_Overtime_Hours']).fillna(0)
        else:
            out['Total_Overtime_Hours'] = (
                safe_num(df.get('Day_Overtime_Hours', 0)) +
                safe_num(df.get('Night_Overtime_Hours', 0))
            )

    # Overtime Ratio
    if 'Overtime_Ratio' in features_list:
        denom = safe_num(df.get('No_of_Working_Days', 1)).replace(0, 1)
        out['Overtime_Ratio'] = out.get('Total_Overtime_Hours', 0) / denom

    # Extra Hours Percentage
    if 'Extra_Hours_Percentage' in features_list:
        typical_day_hours = 8
        denom = safe_num(df.get('No_of_Working_Days', 1)) * typical_day_hours
        denom = denom.replace(0, 1)
        out['Extra_Hours_Percentage'] = out.get('Total_Overtime_Hours', 0) / denom

    # Absenteeism Rate
    if 'Absenteeism_Rate' in features_list:
        dur = safe_num(df.get('Duration', 1)).replace(0, 1)
        out['Absenteeism_Rate'] = safe_num(df.get('Absence_Days', 0)) / dur

    # Tardiness Frequency
    if 'Tardiness_Frequency' in features_list:
        dur = safe_num(df.get('Duration', 1)).replace(0, 1)
        out['Tardiness_Frequency'] = safe_num(df.get('Tardiness_Hours', 0)) / dur

    # Unpaid Leave Rate
    if 'Unpaid_Leave_Rate' in features_list:
        dur = safe_num(df.get('Duration', 1)).replace(0, 1)
        out['Unpaid_Leave_Rate'] = safe_num(df.get('Unpaid_Leave_Days', 0)) / dur

    # Penalty Rate
    if 'Penalty_Rate' in features_list:
        gross = safe_num(df.get('Gross_Salary', 1)).replace(0, 1)
        out['Penalty_Rate'] = safe_num(df.get('Penalty_Value', 0)) / gross

    # Addition Ratio
    if 'Addition_Ratio' in features_list:
        gross = safe_num(df.get('Gross_Salary', 1)).replace(0, 1)
        out['Addition_Ratio'] = safe_num(df.get('Total_Addition_Net', 0)) / gross

    # Deduction Ratio
    if 'Deduction_Ratio' in features_list:
        gross = safe_num(df.get('Gross_Salary', 1)).replace(0, 1)
        out['Deduction_Ratio'] = safe_num(df.get('Total_Deductions', 0)) / gross

    # Years at Company
    if 'Years_At_Company' in features_list:
        dur = safe_num(df.get('Duration', 0))
        out['Years_At_Company'] = dur / 12

    # Dept Avg Salary (fallback)
    if 'Dept_Avg_Salary' in features_list:
        out['Dept_Avg_Salary'] = safe_num(df.get('Gross_Salary', 0))

    if 'Above_Dept_Avg' in features_list:
        out['Above_Dept_Avg'] = 0.0

    # Holiday Work Rate
    if 'Holiday_Work_Rate' in features_list:
        denom = safe_num(df.get('No_of_Working_Days', 1)).replace(0, 1)
        out['Holiday_Work_Rate'] = safe_num(df.get('Holiday_Hours', 0)) / denom

    # Engagement Score (fallback)
    if 'Engagement_Score' in features_list:
        if 'Engagement_Score' in df.columns:
            out['Engagement_Score'] = safe_num(df['Engagement_Score'])
        else:
            out['Engagement_Score'] = 0

    # -------------------------------------------------------
    # FINAL STEP: Ensure *ALL* required features exist
    # -------------------------------------------------------
    for col in features_list:
        if col not in out.columns:
            out[col] = 0.0

    # Reorder to match model input
    out = out[features_list]

    # clean missing values
    out = out.fillna(0).astype(float)

    return out

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
        df_raw = pd.read_csv(uploaded_data)
        st.write("### Preview (Raw Uploaded Data)")
        st.dataframe(df_raw.head())

        # Convert simple CSV → model features
        df_model_ready = preprocess_for_model(df_raw, features)

        st.write("### Engineered Feature Matrix (Model Input)")
        st.dataframe(df_model_ready.head())

        # Predict
        if hasattr(model, "predict_proba"):
            df_raw["Attrition_Probability"] = model.predict_proba(df_model_ready)[:, 1]
        else:
            df_raw["Attrition_Probability"] = model.predict(df_model_ready)

        st.success("Predictions generated successfully!")
        st.dataframe(df_raw)

        st.download_button(
            "📥 Download Predictions",
            data=df_raw.to_csv(index=False),
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
st.info("Model auto-loads from GitHub and preprocessing is applied automatically.")
