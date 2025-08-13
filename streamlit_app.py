import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import gdown

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title='SolarForecastAI', layout='wide')

CSV = "global_solar_data_long.csv"  # file already in your repo
MODEL_PATH = "random_forest.pkl"
FEATURES = [
    "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF", "CLOUD_AMT",
    "T2M", "RH2M", "WS2M"
]
TARGET = "ALLSKY_SFC_SW_DWN"

# Google Drive file ID (replace with your own)
DRIVE_FILE_ID = "16n61Q8I0jaPqkPs6Z7hBGaImSGZdAGS1"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# -----------------------------
# FUNCTIONS
# -----------------------------
@st.cache_data
def load_df():
    return pd.read_csv(CSV)

@st.cache_resource
def load_model():
    # Download model from Google Drive if not already present
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

# -----------------------------
# LOAD DATA & MODEL
# -----------------------------
df = load_df()
model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("☀️ SolarForecastAI — Fast Global Solar Forecast")
st.markdown("Select region → country. Adjust inputs then Predict.")

region_map = {
    "Africa": ["Algeria", "Angola", "Egypt", "Ethiopia", "Gabon", "Ghana", "Kenya", "Morocco", "Nigeria", "South Africa"],
    "Europe": ["France", "Germany", "Italy", "Netherlands", "Poland", "Russia", "Spain", "Ukraine", "United Kingdom"],
    "Asia": ["China", "India", "Indonesia", "Japan", "Malaysia", "Pakistan", "Singapore", "South Korea", "Thailand", "Vietnam"],
    "Americas": ["Argentina", "Brazil", "Canada", "Chile", "Colombia", "Cuba", "Jamaica", "Mexico", "Peru", "United States"],
    "Middle East": ["Israel", "Kuwait", "Oman", "Qatar", "Saudi Arabia", "Turkey", "United Arab Emirates"],
    "Oceania": ["Australia", "Fiji", "New Zealand", "Papua New Guinea"]
}

# Select region and country
region = st.selectbox("Region", list(region_map.keys()))
country = st.selectbox("Country", region_map[region])

# Filter data for the selected country
country_df = df[df["Country"] == country].sort_values("date")
if country_df.empty:
    st.warning("No data for selected country.")
    st.stop()

# Latest row
latest = country_df.iloc[-1]
st.write(f"Latest available data for **{country}**")
st.dataframe(latest[["date"] + FEATURES + [TARGET]].to_frame())

# Input adjustments
st.subheader("Adjust inputs (defaults from latest row)")
cols = st.columns(3)
input_vals = {}
for i, feat in enumerate(FEATURES):
    default = float(latest.get(feat, 0.0))
    input_vals[feat] = cols[i % 3].number_input(feat, value=default)

# Prediction
if st.button("Predict"):
    feat_array = np.array([input_vals[f] for f in FEATURES]).reshape(1, -1)
    pred = model.predict(feat_array)
    st.success(f"Predicted ALLSKY_SFC_SW_DWN: {pred[0]:.4f} kWh/m²")

import streamlit as st
import pandas as pd, joblib, numpy as np, os

st.set_page_config(page_title='SolarForecastAI', layout='wide')

CSV = "global_solar_data_long.csv"
MODELS_DIR = "models"
FEATURES = ["CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "CLOUD_AMT", "T2M", "RH2M", "WS2M"]
TARGET = "ALLSKY_SFC_SW_DWN"

@st.cache_data
def load_df():
    return pd.read_csv(CSV)

@st.cache_resource
def load_models():
    models = {}
    if os.path.exists(os.path.join(MODELS_DIR, "random_forest.pkl")):
        models["Random Forest"] = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    if os.path.exists(os.path.join(MODELS_DIR, "xgboost.pkl")):
        models["XGBoost"] = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
    if os.path.exists(os.path.join(MODELS_DIR, "polynomial_regression.pkl")):
        models["Polynomial Regression"] = joblib.load(os.path.join(MODELS_DIR, "polynomial_regression.pkl"))
    return models

df = load_df()
models = load_models()

st.title("☀️ SolarForecastAI — Fast Global Solar Forecast")
st.markdown("Select region → country → model. Adjust inputs then Predict (no heavy plots for speed).")

region_map = {"Africa": ["Algeria", "Angola", "Egypt", "Ethiopia", "Gabon", "Ghana", "Kenya", "Morocco", "Nigeria", "South Africa", "South Korea"], "Europe": ["France", "Germany", "Italy", "Netherlands", "Poland", "Russia", "Spain", "Ukraine", "United Kingdom"], "Asia": ["China", "India", "Indonesia", "Japan", "Malaysia", "Pakistan", "Singapore", "South Korea", "Thailand", "Vietnam"], "Americas": ["Argentina", "Brazil", "Canada", "Chile", "Colombia", "Cuba", "Jamaica", "Mexico", "Peru", "United States"], "Middle East": ["Israel", "Kuwait", "Oman", "Qatar", "Saudi Arabia", "Turkey", "United Arab Emirates"], "Oceania": ["Australia", "Fiji", "New Zealand", "Papua New Guinea"]}

region = st.selectbox("Region", list(region_map.keys()))
country = st.selectbox("Country", region_map[region])

country_df = df[df["Country"]==country].sort_values("date")
if country_df.empty:
    st.warning("No data for selected country.")
    st.stop()

latest = country_df.iloc[-1]
st.write("Latest available row for", country)
st.dataframe(latest[["date"] + FEATURES + [TARGET]].to_frame())

st.subheader("Adjust inputs (defaults from latest row)")
cols = st.columns(3)
input_vals = {}
for i, feat in enumerate(FEATURES):
    default = float(latest.get(feat, 0.0))
    input_vals[feat] = cols[i % 3].number_input(feat, value=default)

model_choice = st.selectbox("Model", list(models.keys()))
if st.button("Predict"):
    feat_array = np.array([input_vals[f] for f in FEATURES]).reshape(1, -1)
    model = models[model_choice]
    # if xgboost, try scaler
    try:
        if model_choice == "XGBoost" and os.path.exists(os.path.join(MODELS_DIR, "standard_scaler.pkl")):
            scaler = joblib.load(os.path.join(MODELS_DIR, "standard_scaler.pkl"))
            feat_to_pred = scaler.transform(feat_array)
        else:
            feat_to_pred = feat_array
    except Exception:
        feat_to_pred = feat_array
    pred = model.predict(feat_to_pred)
    st.success(f"Predicted ALLSKY_SFC_SW_DWN: {pred[0]:.4f} kWh/m²")
