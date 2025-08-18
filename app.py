"""
Irrigation Recommendation Dashboard

This Streamlit app integrates NOAA and Mesonet APIs to retrieve weather data,
uses an XGBoost model to predict Soil Water Depletion (SWD), and generates
field-level irrigation recommendations. Features include CGDD, ETr, ETa calculations,
precipitation forecasts, and interactive visualizations.

Author: Precious Amori
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
import math
import datetime
import altair as alt
from io import StringIO
from datetime import datetime, timezone

noaa_token = st.secrets["NOAA_TOKEN"]  # Make sure your .streamlit/secrets.toml file contains this

#st.sidebar.subheader("📅 Date Range")
#start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=7))
#end_date = st.sidebar.date_input("End Date", datetime.date.today())

@st.cache_data
def fetch_awdn2_station_list():
    url = "https://awdn2.unl.edu/productdata/get"
    params = {"list": "scqc1440", "network": "nemesonet"}  # daily data for NE Mesonet
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()  # This is a list, not a dict!

    # ✅ Correct way to parse
    stations = [item["name"] for item in data if "name" in item]
    return sorted(stations)


def get_mesonet_weather(station_name, start_date, end_date):
    base_url = "https://awdn2.unl.edu/productdata/get"
    product_id = "scqc1440"  # Daily QC data

    params = (
        f"?name={station_name.replace(' ', '%20')}"
        f"&productid={product_id}"
        f"&begin={start_date}"
        f"&end={end_date}"
        f"&units=si"
        f"&format=csv"
    )
    url = base_url + params
    df = pd.read_csv(url, skiprows=[0, 2])

    # Clean and derive features
    columns_to_drop = [
        'AirTempMax2m_SCQC_Flag','AirTempMin2m_SCQC_Flag','PrecipTotal_SCQC_Flag',
        'RelHumMax2m_SCQC_Flag','RelHumMin2m_SCQC_Flag','SoilTempMax10cm_SCQC_Flag',
        'SoilTempMin10cm_SCQC_Flag','SolarTotal_SCQC_Flag','WindDirectionAvg2m_SCQC_Flag',
        'WindSpeedAvg2m_SCQC_Flag','AtmPressureMax','AtmPressureMax_SCQC_Flag',
        'AtmPressureMin','AtmPressureMin_SCQC_Flag'
    ]
    df_clean = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    df_clean["AvgRH"] = (df_clean["RelHumMax2m"] + df_clean["RelHumMin2m"]) / 2
    df_clean["AvgSoilTemp10cm"] = (df_clean["SoilTempMax10cm"] + df_clean["SoilTempMin10cm"]) / 2

    df_clean = df_clean.drop(columns=[
        "RelHumMax2m", "RelHumMin2m",
        "SoilTempMax10cm", "SoilTempMin10cm",
        "WindDirectionAvg2m"
    ])

    df_clean = df_clean.rename(columns={
        "AirTempMax2m": "T_High_C",
        "AirTempMin2m": "T_Low_C",
        "AvgRH": "Rel Hum %",
        "AvgSoilTemp10cm": "Soil Tmp C@10cm",
        "WindSpeedAvg2m": "Wind Sp. m/s",
        "SolarTotal": "SolarRad MJ/m^2/d",
        "PrecipTotal": "Precip mm",
        "TIMESTAMP": "Date"
    })

    df_clean = df_clean[[
        "Date", "T_High_C", "T_Low_C", "Rel Hum %",
        "Soil Tmp C@10cm", "Wind Sp. m/s", "SolarRad MJ/m^2/d", "Precip mm"
    ]]
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    return df_clean

def fetch_noaa_weather(token, station_id, start_date, end_date):
    url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": token}
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1000,
        "units": "metric"
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        st.warning(f"NOAA API Error {response.status_code}: {response.text}")
        return pd.DataFrame()

    df = pd.DataFrame(response.json().get("results", []))
    if df.empty:
        return df

    wanted = ["TMAX", "TMIN", "PRCP", "AWND", "RHAV"]
    df = df[df['datatype'].isin(wanted)]

    df_pivot = df.pivot_table(index="date", columns="datatype", values="value", aggfunc="first").reset_index()

    rename_map = {
        "date": "Date",
        "TMAX": "T_High_C",
        "TMIN": "T_Low_C",
        "PRCP": "Precip mm",
        "AWND": "Wind Sp. m/s",
        "RHAV": "Rel Hum %"
    }
    df_renamed = df_pivot.rename(columns=rename_map)

    for col in ["T_High_C", "T_Low_C", "Precip mm"]:
        if col in df_renamed.columns:
            df_renamed[col] = df_renamed[col] / 10.0

    df_renamed["Date"] = pd.to_datetime(df_renamed["Date"])
    return df_renamed


def get_noaa_48hr_forecast(lat, lon):
    try:
        # Step 1: Get forecast grid endpoint
        meta_url = f"https://api.weather.gov/points/{lat},{lon}"
        meta_resp = requests.get(meta_url, timeout=10)
        meta_resp.raise_for_status()
        forecast_url = meta_resp.json()["properties"]["forecastHourly"]

        # Step 2: Get hourly forecast
        forecast_resp = requests.get(forecast_url, timeout=10)
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()["properties"]["periods"]

        # Step 3: Convert to DataFrame
        df = pd.DataFrame(forecast_data)

        # Optional: Extract chance of precipitation if available
        if "probabilityOfPrecipitation" in df.columns:
            df["PoP"] = df["probabilityOfPrecipitation"].apply(
                lambda x: x.get("value", 0) if isinstance(x, dict) else 0
            )
        else:
            df["PoP"] = 0

        return df

    except Exception as e:
        print(f"Forecast fetch error: {e}")
        return pd.DataFrame()  # return empty DataFrame to prevent app crash


# ✅ Add these lines right after your imports:
#MODEL_DIR = "models/trained"  # relative to where app.py is located
MODEL_DIR  = os.path.join("models", "trained")
model_path = os.path.join(MODEL_DIR, "XGBoost_vs4.pkl")
scaler_path= os.path.join(MODEL_DIR, "scaler_vs4.pkl")

st.set_page_config(page_title="🌱 Irrigation Dashboard", layout="wide")
st.title("🌱 Irrigation Recommendation Dashboard")

# === Sidebar: model parameters ===
st.sidebar.subheader("📌 Field Information")
field_name = st.sidebar.text_input("Enter Field Name:", value="VRI Field")
field_lat = st.sidebar.number_input("Latitude (°)", value=41.165, format="%.6f")
field_lon = st.sidebar.number_input("Longitude (°)", value=-96.430, format="%.6f")
# Field location parameters for NOAA fallback
#st.sidebar.subheader("🌾 Field Location (for NOAA Fallback)")
#field_lat = st.sidebar.number_input("Field Latitude (°)", value=41.165)
#field_lon = st.sidebar.number_input("Field Longitude (°)", value=-96.430)
st.sidebar.subheader("📥 Upload Raw Dataset for SWD Predictions")
default_url = "https://raw.githubusercontent.com/PreciousAmori/irrigation-dashboard/main/data/ImplementationSET_corn_complete.csv"

raw_file = st.sidebar.file_uploader("Upload Implementation Dataset (CSV):", type=["csv"])
predict_button = st.sidebar.button("🚀 Generate SWD Predictions", key="generate_swd_button")

# If user doesn't upload anything, load the default from GitHub
if raw_file is not None:
    raw_df = pd.read_csv(raw_file)
else:
    try:
        response = requests.get(default_url, timeout=20)
        response.raise_for_status()
        raw_df = pd.read_csv(StringIO(response.text))
        st.info("Using default implementation dataset from GitHub.")
    except Exception as e:
        st.warning(f"Could not load default dataset: {e}")
        raw_df = None

#predict_button = st.sidebar.button("🚀 Generate SWD Predictions")
st.sidebar.header("🔧 Model Parameters")
FC = st.sidebar.number_input("Field Capacity (m³/m³)", value=0.41, step=0.01)
WP = st.sidebar.number_input("Wilting Point (m³/m³)", value=0.19, step=0.01)
f_dc = st.sidebar.slider("Fraction for Maximum Allowable Depletion (MAD fraction)", 0.1, 0.9, 0.5)
# === Sidebar: key dates ===
st.sidebar.header("📅 Key Dates")

emergence_date = st.sidebar.date_input("🌱 Emergence Date (CGDD start):", pd.to_datetime("2024-05-05"))
emergence_date = pd.to_datetime(emergence_date)

date_max = st.sidebar.date_input("📌 Date Max (root depth max / Kcr max):", pd.to_datetime("2024-06-29"))
date_max = pd.to_datetime(date_max)

# 📅 Date Min (automatically one day after emergence)
date_min = emergence_date + pd.Timedelta(days=1)

# Weather station parameters
st.sidebar.subheader("📍 Weather Station Parameters")
elevation = st.sidebar.number_input("Elevation (m)", value=351.0)
latitude_deg = st.sidebar.number_input("Latitude (°)", value=41.15)
longitude_deg = st.sidebar.number_input("Longitude (°)", value=-96.42)
wind_hgt = st.sidebar.number_input("Wind Measurement Height (m)", value=3.0)
albedo = st.sidebar.number_input("Albedo", value=0.23, step=0.01)
Cn = st.sidebar.number_input("Cn", value=1600.0)
Cd = st.sidebar.number_input("Cd", value=0.38)

# NOAA station metadata
noaa_stations = {
    "Mead (USC00256325)": {"id": "GHCND:USC00256325", "lat": 41.165, "lon": -96.430},
    "Mead (alt) (USC00256326)": {"id": "GHCND:USC00256326", "lat": 41.172, "lon": -96.478},
    "Clay Center (USC00251610)": {"id": "GHCND:USC00251610", "lat": 40.580, "lon": -98.129},
    "North Platte (USW00024023)": {"id": "GHCND:USW00024023", "lat": 41.088, "lon": -100.763},
    "Brule (USC00250860)": {"id": "GHCND:USC00250860", "lat": 41.029, "lon": -101.971},
    "Scottsbluff (USW00024045)": {"id": "GHCND:USW00024045", "lat": 41.893, "lon": -103.684},
}

# 🌦️ Weather source section
st.sidebar.subheader("📡 Weather Source")
use_api = st.sidebar.checkbox("Use Mesonet API", value=True)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-05-03"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-09-19"))
start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

noaa_token = st.secrets["NOAA_TOKEN"]

# 🌾 If using Mesonet
if use_api:
    station_list = fetch_awdn2_station_list()
    station_name = st.sidebar.selectbox("Select Weather Station:", station_list)
    st.sidebar.text_input("Fallback NOAA Station ID (optional, used only if Mesonet fails)", key="fallback_noaa_id")

    if st.sidebar.button("📡 Fetch Weather Data"):
        with st.spinner("Fetching weather data..."):
            try:
                weather_df = get_mesonet_weather(station_name, start_str, end_str)
                if weather_df.empty:
                    raise ValueError("Empty Mesonet data returned.")
                st.success(f"✅ Weather data loaded from Mesonet: {station_name}")
            except Exception as e:
                st.warning(f"⚠️ Mesonet fetch failed: {e} — trying fallback NOAA...")
                fallback_id = st.session_state.get("fallback_noaa_id", "")
                if not fallback_id:
                    st.error("❌ No fallback NOAA ID provided.")
                    st.stop()
                try:
                    weather_df = fetch_noaa_weather(noaa_token, fallback_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    if weather_df.empty:
                        st.error("❌ NOAA fetch failed or returned no data.")
                        st.stop()
                    else:
                        st.success(f"✅ NOAA fallback data loaded from {fallback_id}")
                except Exception as ne:
                    st.error(f"❌ NOAA fallback failed: {ne}")
                    st.stop()

        st.session_state["weather_df"] = weather_df

# 🌐 If not using Mesonet — manually fetch NOAA
else:
    st.sidebar.subheader("📡 NOAA Weather Station")
    selected_station = st.sidebar.selectbox(
        "Select a NOAA Station",
        list(noaa_stations.keys()),
        index=None,
        placeholder="Choose a station..."
    )

    fallback_noaa_manual = st.sidebar.text_input("Or manually enter NOAA Station ID (overrides dropdown)", "")

    if selected_station:
        station_meta = noaa_stations[selected_station]
        station_id = station_meta["id"]
        station_lat = station_meta["lat"]
        station_lon = station_meta["lon"]

        st.sidebar.success(f"Selected: {selected_station}")
        st.sidebar.markdown(f"**Station ID:** {station_id}")
        st.sidebar.markdown(f"**Coordinates:** {station_lat:.3f}°, {station_lon:.3f}°")
    else:
        station_id = None

    if st.sidebar.button("📡 Fetch Weather Data"):
        with st.spinner("Fetching NOAA data..."):
            station_to_use = fallback_noaa_manual or station_id
            if not station_to_use:
                st.error("❌ Please select or enter a NOAA station ID.")
                st.stop()
            try:
                weather_df = fetch_noaa_weather(noaa_token, station_to_use, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                if weather_df.empty:
                    st.error("❌ NOAA fetch returned no data.")
                    st.stop()
                else:
                    st.success(f"✅ NOAA data loaded from {station_to_use}")
                    st.session_state["weather_df"] = weather_df
            except Exception as e:
                st.error(f"❌ NOAA API error: {e}")
                st.stop()

        # GDD + ETr calculation block for any weather_df
        if weather_df is not None:
            st.session_state["weather_df"] = weather_df  # Save to session    

# === File uploaders ===
pred_file = st.sidebar.file_uploader("📄 Upload SWD Predictions CSV", type=["csv"])
weather_file = st.sidebar.file_uploader("☀️ Upload Weather CSV", type=["csv"])

# === Optional: Generate predictions from raw file ===
#raw_file = st.sidebar.file_uploader("📥 Upload Implementation Dataset (CSV):", type=["csv"])
#predict_button = st.sidebar.button("🚀 Generate SWD Predictions", key="generate_swd_button")

#if predict_button and raw_df is not None:
#    id_df = raw_df[['Date','Management Plot ID']].copy()
#    X_features = raw_df.drop(columns=['Date'], errors='ignore')

    # ✅ Load model & scaler
#    model = joblib.load(model_path)
#    scaler = joblib.load(scaler_path)

    # ✅ Predict
#    X_scaled = scaler.transform(X_features)
#    predictions = model.predict(X_scaled)

#    predictions_df = id_df.copy()
#    predictions_df['SWD_predictions'] = predictions

#    predictions_df.rename(columns={"Management Plot ID": "Management_Plot_ID"}, inplace=True)

    # ✅ Store in session_state so it persists
#    st.session_state['pred_df'] = predictions_df

#    st.success("✅ SWD Predictions generated successfully! Scroll down to proceed.")

    # 🔥 Feed predictions_df into the rest of the pipeline
#    pred_df = predictions_df.copy()

# --- Predict on click, using raw_df (uploaded OR default) ---
# --- Predict on click, using raw_df (uploaded OR default) ---
if predict_button and raw_df is not None:
    # Required columns
    if not {'Date', 'Management Plot ID'}.issubset(raw_df.columns):
        st.error("Dataset must contain 'Date' and 'Management Plot ID'.")
        st.stop()

    # Ensure types
    raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')
    raw_df['Management Plot ID'] = pd.to_numeric(raw_df['Management Plot ID'], errors='coerce')

    # IDs for output
    id_df = raw_df[['Date', 'Management Plot ID']].copy()

    # ✅ Features: drop ONLY Date (keep MPID); keep numeric; fill NAs
    X_features = (
        raw_df.drop(columns=['Date'], errors='ignore')
              .select_dtypes(include=[np.number])
              .fillna(0.0)
    )

    # Load artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Align feature order if the scaler exposes it
    if hasattr(scaler, "feature_names_in_"):
        expected = pd.Index(scaler.feature_names_in_)
        orig_cols = X_features.columns
        # Reindex to expected order; fill any missing columns with 0.0 and drop extras
        X_features = X_features.reindex(columns=expected, fill_value=0.0)

    # Transform & predict
    X_scaled = scaler.transform(X_features)
    predictions = model.predict(X_scaled)

    # Output
    predictions_df = id_df.rename(columns={"Management Plot ID": "Management_Plot_ID"})
    predictions_df['SWD_predictions'] = predictions
    st.session_state['pred_df'] = predictions_df
    st.success("✅ SWD Predictions generated successfully! Scroll down to proceed.")
    pred_df = predictions_df.copy()

# Check if predictions are generated OR a CSV is uploaded
if 'pred_df' not in st.session_state:
    if pred_file is not None:
        st.session_state['pred_df'] = pd.read_csv(pred_file)

# Use pred_df from session_state for pipeline
if 'pred_df' in st.session_state and 'weather_df' in st.session_state:
    pred_df = st.session_state['pred_df'].copy()
    weather_df = st.session_state['weather_df'].copy()

    # Clean column names
    weather_df.columns = weather_df.columns.str.strip()

    # Ensure Date columns
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])

    # === GDD and CGDD calculations ===
    Tbase = 10.0
    Tmax_cap = 30.0
    weather_df['Tmax_Lim'] = weather_df['T_High_C'].clip(lower=Tbase, upper=Tmax_cap)
    weather_df['Tmin_Lim'] = weather_df['T_Low_C'].clip(lower=Tbase, upper=Tmax_cap)
    weather_df['Tavg'] = (weather_df['T_High_C'] + weather_df['T_Low_C']) / 2
    weather_df['GDD'] = ((weather_df['Tmax_Lim'] + weather_df['Tmin_Lim']) / 2) - Tbase
    weather_df['GDD'] = weather_df['GDD'].clip(lower=0)

    #emergence_date = pd.to_datetime("2024-05-05")
    weather_df = weather_df.sort_values('Date')
    weather_df['CGDD'] = 0.0
    mask = weather_df['Date'] >= emergence_date
    weather_df.loc[mask, 'CGDD'] = weather_df.loc[mask, 'GDD'].cumsum()

    # === ETr calculations ===
    phi = math.pi * latitude_deg / 180
    Gsc = 4.92
    sigma = 4.901e-9
    G = 0.0
    weather_df['doy'] = weather_df['Date'].dt.dayofyear
    weather_df['es'] = (0.6108 * np.exp((17.27 * weather_df['T_High_C'])/(weather_df['T_High_C']+237.3)) +
                        0.6108 * np.exp((17.27 * weather_df['T_Low_C'])/(weather_df['T_Low_C']+237.3))) / 2
    weather_df['ea'] = (weather_df['Rel Hum %'] / 100.0) * weather_df['es']
    weather_df['dr'] = 1 + 0.033 * np.cos(2 * np.pi * weather_df['doy'] / 365)
    weather_df['delta'] = 0.409 * np.sin((2 * np.pi * weather_df['doy'] / 365) - 1.39)
    weather_df['ws'] = np.arccos(-np.tan(phi) * np.tan(weather_df['delta']))
    weather_df['Ra'] = (24/np.pi * Gsc * weather_df['dr'] *
                        (weather_df['ws']*np.sin(phi)*np.sin(weather_df['delta']) +
                         np.cos(phi)*np.cos(weather_df['delta'])*np.sin(weather_df['ws'])))
    # === Estimate solar radiation Rs (MJ/m^2/day) using Hargreaves-Samani if missing ===
    if 'SolarRad MJ/m^2/d' not in weather_df.columns:
        if latitude_deg < 20:
            kRs = 0.19  # coastal
        else:
            kRs = 0.16  # inland

        st.info(f"☀️ Estimating solar radiation using Hargreaves method with kRs={kRs}")

        weather_df['SolarRad MJ/m^2/d'] = (
            kRs * np.sqrt(weather_df['T_High_C'] - weather_df['T_Low_C']) * weather_df['Ra']
        )
        weather_df['SolarRad MJ/m^2/d_source'] = 'Estimated (Hargreaves)'
    else:
        weather_df['SolarRad MJ/m^2/d_source'] = 'Observed'

    weather_df['Rso'] = (0.75 + 2e-5 * elevation) * weather_df['Ra']

    # === Handle missing Solar Radiation column for NOAA fallback ===
    #if "SolarRad MJ/m^2/d" not in weather_df.columns:
        #st.warning("⚠️ 'SolarRad MJ/m^2/d' not found in NOAA data — ETr will be NaN.")
        #weather_df["SolarRad MJ/m^2/d"] = np.nan
    weather_df['fcd'] = 1.35 * np.minimum(np.maximum(weather_df['SolarRad MJ/m^2/d'] / weather_df['Rso'], 0.3), 1.0) - 0.35
    weather_df['Rns'] = (1 - albedo) * weather_df['SolarRad MJ/m^2/d']
    weather_df['Rnl'] = sigma * weather_df['fcd'] * (0.34 - 0.14*np.sqrt(weather_df['ea'])) * (
        ((weather_df['Tmax_Lim']+273.16)**4 + (weather_df['Tmin_Lim']+273.16)**4)/2 )
    weather_df['Rn'] = weather_df['Rns'] - weather_df['Rnl']
    P = 101.3 * (((293 - 0.0065*elevation)/293)**5.26)
    gamma = 0.000665 * P
    weather_df['delta_slope'] = (2503*np.exp((17.27*weather_df['Tavg'])/(weather_df['Tavg']+237.3))) / ((weather_df['Tavg']+237.3)**2)
    weather_df['u2'] = weather_df['Wind Sp. m/s'] * 4.87 / np.log(67.8 * wind_hgt - 5.42)
    weather_df['ETr'] = (
        (0.408 * weather_df['delta_slope'] * (weather_df['Rn'] - G)
         + gamma * (Cn / (weather_df['Tavg'] + 273.0)) * weather_df['u2'] * (weather_df['es'] - weather_df['ea']))
        / (weather_df['delta_slope'] + gamma * (1.0 + Cd * weather_df['u2']))
    )

    # === Merge with predictions
    df = pred_df.merge(weather_df[['Date', 'GDD', 'CGDD', 'ETr']], on='Date', how='left')

    # === Compute Kcr
    def compute_kcr(cgdd):
        if cgdd < 100: return 0.4
        elif cgdd < 600: return 0.4 + 0.0012*(cgdd-100)
        elif cgdd < 1400: return 1.0
        elif cgdd < 1700: return 1.0 + (-0.0027)*(cgdd-1400)
        else: return 0.2
    df['Kcr'] = df['CGDD'].apply(compute_kcr)
    df['ETa'] = df['ETr'] * df['Kcr']

    # === Water balance (MAD, TD, Recommended irrigation)
    #date_min = pd.to_datetime("2024-05-06")
    #date_max = pd.to_datetime("2024-06-29")
    def calc_root_depth(d):
        if d <= date_min: return 100
        elif d >= date_max: return 1000
        else:
            days_total = (date_max - date_min).days
            days_elapsed = (d - date_min).days
            return 100 + (1000 - 100) * (days_elapsed / days_total)
    df['RootDepth_mm'] = df['Date'].apply(calc_root_depth)
    df['TAW_mm'] = df['RootDepth_mm'] * (FC - WP)
    df['MAD'] = df['TAW_mm'] * f_dc
    df['TD'] = (df['MAD'] - 25.9).clip(lower=0)
    df['Recommended_Irrigation_mm'] = (df['SWD_predictions'] - df['TD']).clip(lower=0)
    irrigation_recommended = df['Recommended_Irrigation_mm'].iloc[-1] > 0


    # Optional forecast suggestion based on irrigation need
    if irrigation_recommended:
        st.warning("⚠️ Irrigation is recommended. Consider checking upcoming rainfall forecasts.")

        if st.checkbox("🔍 View 48-hour NOAA Rainfall Forecast", key="forecast_toggle"):
            forecast_df = get_noaa_48hr_forecast(field_lat, field_lon)

            if forecast_df.empty:
                st.error("❌ Forecast could not be loaded. Please check internet connection or NOAA API.")
            else:
                # Convert time and calculate forecast hour
                forecast_df["time"] = pd.to_datetime(forecast_df["startTime"])
                forecast_df["Hour"] = range(len(forecast_df))

                now_utc = datetime.now(timezone.utc)
                forecast_start = forecast_df["time"].iloc[0]
                current_hour_index = int((now_utc - forecast_start).total_seconds() // 3600)

                # Create Altair chart
                line = alt.Chart(forecast_df).mark_line(point=True).encode(
                    x=alt.X("Hour:Q", title="Forecast Hour (0–155)"),
                    y=alt.Y("PoP:Q", title="Precipitation Probability (%)"),
                    tooltip=["time:T", "shortForecast:N", "PoP:Q"]
                ).properties(title="🌧️ 48-Hour NOAA Rainfall Forecast")

                now_rule = alt.Chart(pd.DataFrame({
                    "Hour": [current_hour_index]
                })).mark_rule(color="red", strokeDash=[5, 5], strokeWidth=3).encode(x="Hour:Q")

                st.altair_chart((line + now_rule).interactive(), use_container_width=True)

                # Table view of the forecast
                forecast_df["Hour"] = range(len(forecast_df))  # Add this line
                st.dataframe(forecast_df[["Hour", "shortForecast", "PoP"]].head(10).rename(columns={
                    "shortForecast": "Forecast",
                    "PoP": "Precipitation Probability (%)"
                }))



    # === Dashboard
    plot_ids = df['Management_Plot_ID'].unique()
    selected_id = st.selectbox("Select a Plot ID:", plot_ids)
    filtered = df[df['Management_Plot_ID'] == selected_id]

    # ✅ Date range slider
    min_date = filtered['Date'].min().date()
    max_date = filtered['Date'].max().date()
    date_range = st.slider(
        "📅 Select Date Range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="MM/DD/YYYY"
    )

    # Filter by selected date range
    filtered = filtered[(filtered['Date'].dt.date >= date_range[0]) & (filtered['Date'].dt.date <= date_range[1])]

    # ✅ Metrics selector
    metrics = st.multiselect(
        "📊 Select variables to plot:",
        ["SWD_predictions", "MAD", "TD", "Kcr", "ETr", "ETa"],
        default=["SWD_predictions"]
    )

    # ✅ Melt the data for selected metrics into long format
    melted = filtered.melt(
        id_vars=['Date'],
        value_vars=metrics,
        var_name='Variable',
        value_name='Value'
    )

    # ✅ Define color mapping with legend-friendly colors
    color_map = {
        'SWD_predictions': 'blue',
        'MAD': 'red',
        'TD': 'green',
        'Kcr': 'purple',
        'ETr': 'gray',       # changed to gray
        'ETa': 'brown'
    }

    # ✅ Build base chart with short date format and legend
    line_chart = alt.Chart(melted).mark_line(strokeWidth=2).encode(
        x=alt.X(
            'Date:T',
            title='Date',
            axis=alt.Axis(format='%m/%d', labelAngle=-45, labelFontSize=12, titleFontSize=14)
        ),
        y=alt.Y(
            'Value:Q',
            title='Values (mm or unit)',
            axis=alt.Axis(labelFontSize=12, titleFontSize=14)
        ),
        color=alt.Color(
            'Variable:N',
            scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
            legend=alt.Legend(title="Variables", labelFontSize=12, titleFontSize=14, orient='top', direction='horizontal')
        ),
        tooltip=['Date:T', 'Variable:N', 'Value:Q']
    ).properties(
        width=1500,
        height=450,
        #title=f"Selected Metrics for Plot {selected_id}"
        #title=f"Selected Variables for {field_name} – Plot {selected_id}"
    )

    # ✅ Merge predicted irrigation and weather precipitation
    merged_df = pd.merge(
        filtered[['Date', 'Recommended_Irrigation_mm']],
        weather_df[['Date', 'Precip mm']],
        on='Date',
        how='inner'  # Use 'outer' if needed
    )

    # ✅ Melt for combined chart with unified y-axis
    plot_df = merged_df.melt(id_vars='Date', var_name='Metric', value_name='Value')

    # ✅ Define chart components
    color_scale = alt.Scale(domain=['Recommended_Irrigation_mm', 'Precip mm'],
                            range=['orange', '#2B9CE2'])

    # ✅ Separate bar and line marks by condition
    bars = alt.Chart(plot_df[plot_df['Metric'] == 'Recommended_Irrigation_mm']).mark_bar(
        opacity=0.7
    ).encode(
        x=alt.X('Date:T', axis=alt.Axis(format='%m/%d', labelAngle=-45)),
        y=alt.Y('Value:Q', title='Depth of water (mm)'),
        color=alt.value('orange'),
        tooltip=['Date:T', 'Metric:N', 'Value:Q']
    )

    lines = alt.Chart(plot_df[plot_df['Metric'] == 'Precip mm']).mark_line(
        strokeWidth=2,
        color='#2B9CE2'
    ).encode(
        x='Date:T',
        y='Value:Q',
        tooltip=['Date:T', 'Metric:N', 'Value:Q']
    )

    # ✅ Overlay chart with shared y-axis
    combined_chart = alt.layer(bars, lines).properties(
        width=1500,
        height=400,
        title='Irrigation Recommendations with Precipitation Overlay'
    )

    # ✅ Display charts
    st.altair_chart(line_chart, use_container_width=True)
    st.altair_chart(combined_chart, use_container_width=True)

    # ✅ Show rounded table
    st.write("### 📑 Recommendations Table")
    cols_to_move = ["GDD", "CGDD", "ETr", "Kcr", "ETa", "Recommended_Irrigation_mm"]
    other_cols = [c for c in filtered.columns if c not in cols_to_move]
    ordered = filtered[other_cols + cols_to_move]
    st.dataframe(ordered.round(1))

    st.download_button(
        "📥 Download CSV",
        data=ordered.to_csv(index=False).encode('utf-8'),
        #file_name="Irrigation_Recommendations.csv",
        file_name=f"Irrigation_Recommendations_{field_name.replace(' ','_')}.csv",
        mime='text/csv'
    )
