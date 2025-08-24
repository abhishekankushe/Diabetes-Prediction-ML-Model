# app_streamlit.py — auto-predict on input change, no button
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- MODEL UTILS ---------
zero_as_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def zeros_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in zero_as_missing_cols:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    return df

@st.cache_resource
def load_pipeline():
    # Ensure the .joblib file is in the same directory or adjust the path
    return joblib.load('diabetes_decision_tree_pipeline.joblib')

pipe = load_pipeline()

# ---------- UI ----------
st.title("Diabetes Risk Prediction")
st.caption("Adjust inputs below. The prediction updates automatically on any change.")

# Defaults
default_vals = {
    "Pregnancies": 1,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 85,
    "BMI": 30.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}

col1, col2 = st.columns(2)
with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=default_vals["Pregnancies"], key="preg")
    Glucose = st.number_input("Glucose", min_value=0, value=default_vals["Glucose"], key="glu")
    BloodPressure = st.number_input("BloodPressure", min_value=0, value=default_vals["BloodPressure"], key="bp")
    SkinThickness = st.number_input("SkinThickness", min_value=0, value=default_vals["SkinThickness"], key="skin")
with col2:
    Insulin = st.number_input("Insulin", min_value=0, value=default_vals["Insulin"], key="ins")
    BMI = st.number_input("BMI", min_value=0.0, format="%.1f", value=default_vals["BMI"], key="bmi")
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, format="%.3f",
                                               value=default_vals["DiabetesPedigreeFunction"], key="dpf")
    Age = st.number_input("Age", min_value=0, step=1, value=default_vals["Age"], key="age")

# Build input frame from current widget values
input_df = pd.DataFrame([{
    "Pregnancies": Pregnancies,
    "Glucose": Glucose,
    "BloodPressure": BloodPressure,
    "SkinThickness": SkinThickness,
    "Insulin": Insulin,
    "BMI": BMI,
    "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
    "Age": Age
}])

# Optional: show current inputs for transparency/debug
with st.expander("Show current input row", expanded=False):
    # style: bigger font, some padding, center alignment
    styled = (
        input_df.style
        .set_properties(**{
            "font-size": "30px",      # increase text size
            "padding": "20px 25px",   # bigger cells
            "text-align": "center"
        })
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "30px"), ("padding", "20px 25px")]},
            {"selector": "tbody td", "props": [("font-size", "30px"), ("padding", "20px 25px")]}
        ])
    )

    st.dataframe(styled, height=130, use_container_width=True)



# Predict automatically on every rerun
try:
    pred = int(pipe.predict(input_df)[0])
    proba = float(pipe.predict_proba(input_df)[0, 1])

    if pred == 1:
        st.error(f"High risk of Diabetes (Probability: {proba:.1%})")
    else:
        st.success(f"Low risk of Diabetes (Probability: {proba:.1%})")
except Exception as e:
    st.warning("Prediction could not be computed. Check console for details.")
    st.exception(e)

st.divider()
st.subheader("Batch prediction via CSV")
st.caption("Upload CSV with columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age")

file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    df = pd.read_csv(file)

    feature_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
    # Align columns; warn if missing
    
    
    # keep only expected features; auto-drop extras like Outcome/ID/etc.
    df_features = df.reindex(columns=feature_cols)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing required columns in CSV: {', '.join(missing)}")

    preds = pipe.predict(df_features)
    probas = pipe.predict_proba(df_features)[:, 1]
    out = df.copy()
    out["Pred"] = preds
    out["Prob_Diabetes"] = probas

    st.dataframe(out.head(500))
    csv = out.to_csv(index=False).encode()
    st.download_button("Download predictions", csv, "predictions.csv", "text/csv")

st.markdown('</div>', unsafe_allow_html=True)
