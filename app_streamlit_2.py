# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from pathlib import Path

# ---------- THEME / CSS  ----------
BG_IMAGE_URL = "assets/bg.jpg"  # local path OR a remote https:// URL

def inject_css(bg_path_or_url: str):
    def _encode_local(path: str) -> str:
        p = Path(path)
        if not p.exists():
            return ""
        return base64.b64encode(p.read_bytes()).decode()

    is_remote = str(bg_path_or_url).startswith("http")
    bg_css = ""
    if is_remote:
        bg_css = f"url('{bg_path_or_url}')"
    else:
        b64 = _encode_local(bg_path_or_url)
        if b64:
            bg_css = f"url('data:image/jpg;base64,{b64}')"

    st.markdown(
        f"""
        <style>
        :root {{
          --brand: #2563eb;            /* blue-600 */
          --brand-600: #2563eb;
          --brand-700: #1d4ed8;
          --text: #0f172a;             /* slate-900 */
          --muted: #64748b;            /* slate-500 */
          --card: rgba(255,255,255,0.75);
          --glass: rgba(255,255,255,0.6);
          --border: rgba(15,23,42,0.08);
        }}

        /* page background */
        [data-testid="stAppViewContainer"] {{
          background-image: {bg_css if bg_css else "none"};
          background-size: cover;
          background-position: center;
        }}

        /* subtle gradient overlay for readability */
        [data-testid="stAppViewContainer"]::before {{
          content: "";
          position: fixed;
          inset: 0;
          background: radial-gradient(1200px 600px at 20% 10%, rgba(255,255,255,0.75), transparent 60%),
                      linear-gradient(to bottom right, rgba(255,255,255,0.75), rgba(255,255,255,0.35));
          pointer-events: none;
          z-index: 0;
        }}

        /* header + sidebar polish */
        [data-testid="stHeader"] {{
          background: transparent;
        }}
        section[data-testid="stSidebar"] > div:first-child {{
          background: rgba(255,255,255,0.85);
          backdrop-filter: saturate(160%) blur(8px);
          border-right: 1px solid var(--border);
        }}

        /* Tailwind-like utility classes (mimic) */
        .tw-container {{
          position: relative;
          z-index: 1;
          padding: 24px;
          border-radius: 18px;
          background: var(--glass);
          backdrop-filter: saturate(180%) blur(12px);
          box-shadow: 0 10px 30px rgba(2,6,23,0.12);
          border: 1px solid var(--border);
        }}
        .tw-card {{
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 16px 18px;
          box-shadow: 0 6px 18px rgba(2,6,23,0.08);
        }}
        .tw-title {{
          font-size: 28px;
          font-weight: 800;
          letter-spacing: -0.02em;
          color: var(--text);
          margin-bottom: 6px;
        }}
        .tw-sub {{
          color: var(--muted);
          margin-bottom: 18px;
        }}
        .tw-chip {{
          display: inline-block;
          padding: 6px 10px;
          background: rgba(37,99,235,0.12);
          color: var(--brand-700);
          border: 1px solid rgba(37,99,235,0.22);
          border-radius: 999px;
          font-weight: 600;
          font-size: 12px;
          margin-left: 8px;
        }}

        /* buttons */
        .stButton > button {{
          background: var(--brand-600);
          color: white;
          border: none;
          border-radius: 12px;
          padding: 10px 16px;
          font-weight: 700;
          box-shadow: 0 8px 20px rgba(37,99,235,0.35);
          transition: transform .04s ease, box-shadow .2s ease, background .2s ease;
        }}
        .stButton > button:hover {{
          background: var(--brand-700);
          transform: translateY(-1px);
          box-shadow: 0 10px 24px rgba(37,99,235,0.45);
        }}

        /* inputs */
        .stNumberInput, .stTextInput, .stSelectbox, .stFileUploader {{
          border-radius: 12px !important;
        }}
        [data-baseweb="input"] > div {{
          border-radius: 12px !important;
        }}

        /* tables */
        .stDataFrame, .stTable {{
          border-radius: 12px;
          overflow: hidden;
          background: rgba(255,255,255,0.92);
          border: 1px solid var(--border);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# call CSS early
inject_css(BG_IMAGE_URL)

# ---------- MODEL UTILS (same as before) ----------
zero_as_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def zeros_to_nan(df):
    df = df.copy()
    for c in zero_as_missing_cols:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    return df

@st.cache_resource
def load_pipeline():
    return joblib.load('diabetes_decision_tree_pipeline.joblib')

pipe = load_pipeline()

# ---------- UI ----------
st.markdown('<div class="tw-container">', unsafe_allow_html=True)
st.markdown(
    '<div class="tw-title">🩺 Diabetes Risk Prediction <span class="tw-chip">Decision Tree</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="tw-sub">Enter patient data (or upload CSV) to predict diabetes (0 = No, 1 = Yes).</div>',
    unsafe_allow_html=True
)

# feature inputs (unchanged)
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
    with st.container():
        st.markdown('<div class="tw-card">', unsafe_allow_html=True)
        Pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=default_vals["Pregnancies"])
        Glucose = st.number_input("Glucose", min_value=0, value=default_vals["Glucose"])
        BloodPressure = st.number_input("BloodPressure", min_value=0, value=default_vals["BloodPressure"])
        SkinThickness = st.number_input("SkinThickness", min_value=0, value=default_vals["SkinThickness"])
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="tw-card">', unsafe_allow_html=True)
        Insulin = st.number_input("Insulin", min_value=0, value=default_vals["Insulin"])
        BMI = st.number_input("BMI", min_value=0.0, format="%.1f", value=default_vals["BMI"])
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, format="%.3f",
                                                   value=default_vals["DiabetesPedigreeFunction"])
        Age = st.number_input("Age", min_value=0, step=1, value=default_vals["Age"])
        st.markdown('</div>', unsafe_allow_html=True)

if st.button(" Predict single case"):
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
    pred = pipe.predict(input_df)[0]
    proba = float(pipe.predict_proba(input_df)[0, 1])
    if pred == 1:
        st.error(f" High risk of Diabetes (Probability: {proba:.1%})")
    else:
        st.success(f" Low risk of Diabetes (Probability: {proba:.1%})")

st.markdown("---")
st.subheader("Batch prediction via CSV")

st.caption("Upload CSV with columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age")
file = st.file_uploader("Upload CSV", type=['csv'])
if file is not None:
    df = pd.read_csv(file)

    feature_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
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
