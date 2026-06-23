"""
Irrigation Recommendation Dashboard

This Streamlit app integrates Mesonet (agreport) and NOAA weather data, as well as other features,
uses an XGBoost model to predict Soil Water Depletion (SWD), and generates
management plot-level irrigation recommendations. It supports chunked agreport fetches
with timezone fallback, manual weather and daily model input CSV upload, CGDD/ETr/ETa calculations,
precipitation forecasts, and interactive visualizations.

Author: Precious Amori
"""

# =========================
# Imports
# =========================
import io
import os
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import joblib
import altair as alt
import streamlit as st

# =========================
# Secrets / Paths
# =========================
noaa_token = st.secrets["NOAA_TOKEN"]  # NOAA stays the same

# =========================
# Model paths (use in-repo copies with underscore names)
# =========================
MODEL_DIR = os.path.join("models", "trained")
model_path = os.path.join(MODEL_DIR, "XGB_loso_scal.joblib")


# =========================
# Streamlit page setup
# =========================
st.set_page_config(page_title="🌱 Irrigation Dashboard", layout="wide")
st.title("🌱 Irrigation Recommendation Dashboard - Phase 2")

# ============================================================
# Mesonet helpers (NEW — agreport + chunking, metric output)
# ============================================================
import io
from urllib.parse import quote

BASE = "https://awdn.unl.edu/productdata/get"

def _ymd(d) -> str:
    return pd.to_datetime(d).strftime("%Y%m%d")

def _build_url(name: str, end: str, days: int, tz: str, units: str = "si", fmt: str = "csv") -> str:
    return (
        f"{BASE}"
        f"?name={quote(name)}"
        f"&productid=agreport"
        f"&end={pd.to_datetime(end).strftime('%Y%m%d')}"
        f"&days={int(days)}"
        f"&tz={tz}"
        f"&units={units}"
        f"&format={fmt}"
    )

def _fetch_agreport_chunk(
    name: str,
    end: str,
    days: int,
    tz: str,
    connect_timeout: int = 10,
    read_timeout: int = 120,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Fetch one Mesonet 'agreport' chunk. If the provided tz triggers a 4xx (e.g., 400),
    transparently retry with known-good fallbacks ('EST', then 'UTC').
    """
    def _try_with_tz(tz_try: str) -> pd.DataFrame:
        url = _build_url(name, end, days, tz_try)
        if verbose:
            st.write(f"[GET] {url}")
        r = requests.get(url, timeout=(connect_timeout, read_timeout))
        r.raise_for_status()
        # Mesonet CSV typically has metadata rows at [0] and [2]
        return pd.read_csv(io.StringIO(r.text), skiprows=[0, 2])

    # Try the requested tz first, then fallbacks
    tz_candidates = [tz] + ([t for t in ["EST", "UTC"] if t != tz])
    last_err = None

    for tz_try in tz_candidates:
        try:
            df = _try_with_tz(tz_try)
            if tz_try != tz and verbose:
                st.info(f"Mesonet tz '{tz}' failed, used fallback '{tz_try}'.")
            return df
        except requests.HTTPError as e:
            # 4xx/5xx → try next candidate
            last_err = e
        except Exception as e:
            last_err = e

    # If all attempts failed, bubble up the last error
    raise last_err if last_err else RuntimeError("Mesonet request failed for all tz candidates.")


import math  # ⟵ add near your imports

@st.cache_data(show_spinner=False)
def fetch_agreport_chunked(name: str, start_date, end_date, tz: str = "CST6CDT",
                           chunk_days: int = 10, connect_timeout: int = 10, read_timeout: int = 120,
                           verbose: bool = False, show_progress: bool = False) -> pd.DataFrame:
    """
    Pull Mesonet 'agreport' in chunks (end+days windows), stitch into one DataFrame.
    Returns the raw (US-units) agreport table exactly as Mesonet sends it.
    """
    start = pd.to_datetime(start_date).normalize()
    end   = pd.to_datetime(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be >= start_date")

    # --- Progress setup (accurate by number of chunks) ---
    total_days   = (end - start).days + 1
    total_chunks = max(1, math.ceil(total_days / max(1, int(chunk_days))))
    progress = st.progress(0) if show_progress else None
    done = 0
    # ------------------------------------------------------

    out = []
    # Walk backwards in inclusive 10-day windows (or chunk_days)
    cur_end = end
    while cur_end >= start:
        # Window start for this chunk
        cur_start = max(start, cur_end - pd.Timedelta(days=chunk_days - 1))
        days = (cur_end - cur_start).days + 1
        try:
            dfc = _fetch_agreport_chunk(name, cur_end, days, tz,
                                        connect_timeout=connect_timeout,
                                        read_timeout=read_timeout,
                                        verbose=verbose)
            out.append(dfc)
        except Exception as e:
            if verbose:
                st.warning(f"[CHUNK ERROR] {cur_start.date()}–{cur_end.date()}: {e}")
            raise
        finally:
            # advance progress whether success or failure
            done += 1
            if progress:
                progress.progress(min(done / total_chunks, 1.0))

        # Next window ends the day before this chunk’s start
        cur_end = cur_start - pd.Timedelta(days=1)

    if progress:
        progress.empty()

    if not out:
        return pd.DataFrame()

    df_raw = pd.concat(out, ignore_index=True) 


    # Normalize headers / dates / dedupe by day (overlap between chunks is expected)
    df_raw.columns = df_raw.columns.str.strip()
    if "Timestamp" in df_raw.columns:
        df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"], errors="coerce").dt.date
        df_raw = (
            df_raw.dropna(subset=["Timestamp"])
                  .drop_duplicates(subset=["Timestamp"], keep="last")
                  .sort_values("Timestamp")
                  .reset_index(drop=True)
        )
    # Trim to requested window (belt & suspenders)
    if "Timestamp" in df_raw.columns:
        m = (pd.to_datetime(df_raw["Timestamp"]) >= start) & (pd.to_datetime(df_raw["Timestamp"]) <= end)
        df_raw = df_raw.loc[m].reset_index(drop=True)

    return df_raw

def to_metric_8cols(df_ag: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the Ag Report (US units) to the 8-column metric schema:
    Date, T_High_C, T_Low_C, Rel Hum %, Soil Tmp C@10cm, Wind Sp. m/s, SolarRad MJ/m^2/d, Precip mm
    """
    # Soft rename from Mesonet agreport headers
    ren = {
        "Timestamp": "Date",
        "Max Temperature (F)": "Tmax_F",
        "Min Temperature (F)": "Tmin_F",
        "Relative Humidity (%)": "RH_pct",
        "Soil Temperature at 4 inches (F)": "SoilTemp4in_F",
        "Solar Radiation (MJ/m^2/day)": "Solar_MJm2d",
        "Precipitation (in)": "Precip_in",
        "Wind Speed (mph)": "Wind_mph",
    }
    df = df_ag.rename(columns={k: v for k, v in ren.items() if k in df_ag.columns}).copy()

    need = ["Date","Tmax_F","Tmin_F","RH_pct","SoilTemp4in_F","Solar_MJm2d","Precip_in","Wind_mph"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Ag Report missing expected columns: {miss}")

    # Unit conversions
    f2c    = lambda f: (pd.to_numeric(f, errors="coerce") - 32.0) * (5.0/9.0)
    mph2ms = lambda v: pd.to_numeric(v, errors="coerce") * 0.44704
    in2mm  = lambda x: pd.to_numeric(x, errors="coerce") * 25.4

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    out["T_High_C"]            = f2c(df["Tmax_F"])
    out["T_Low_C"]             = f2c(df["Tmin_F"])
    out["Rel Hum %"]           = pd.to_numeric(df["RH_pct"], errors="coerce")
    out["Soil Tmp C@10cm"]     = f2c(df["SoilTemp4in_F"])      # 4 in ≈ 10.16 cm
    out["Wind Sp. m/s"]        = mph2ms(df["Wind_mph"])
    out["SolarRad MJ/m^2/d"]   = pd.to_numeric(df["Solar_MJm2d"], errors="coerce")  # already MJ/m²/day
    out["Precip mm"]           = in2mm(df["Precip_in"])

    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Match your previous formatting
    out = out.round({
        "T_High_C": 2, "T_Low_C": 2, "Rel Hum %": 2, "Soil Tmp C@10cm": 3,
        "Wind Sp. m/s": 2, "SolarRad MJ/m^2/d": 2, "Precip mm": 2
    })

    return out[["Date","T_High_C","T_Low_C","Rel Hum %","Soil Tmp C@10cm","Wind Sp. m/s","SolarRad MJ/m^2/d","Precip mm"]]

@st.cache_data(show_spinner=False)
def get_mesonet_weather(name: str, start_date, end_date, tz: str = "CST6CDT",
                        chunk_days: int = 10, connect_timeout: int = 10, read_timeout: int = 120,
                        verbose: bool = False, show_progress: bool = False) -> pd.DataFrame:
    # fetch raw agreport chunks (US units)
    df_raw = fetch_agreport_chunked(
        name, start_date, end_date, tz=tz,
        chunk_days=chunk_days,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        verbose=verbose,
        show_progress=show_progress,  # <-- added
    )
    if df_raw.empty:
        return df_raw

    # convert to your 8-column metric schema
    df_clean = to_metric_8cols(df_raw)

    # final guard: clamp to [start, end]
    s = pd.to_datetime(start_date).normalize()
    e = pd.to_datetime(end_date).normalize()
    df_clean = df_clean[(df_clean["Date"] >= s) & (df_clean["Date"] <= e)].reset_index(drop=True)
    return df_clean


# ============================================================
# NOAA daily (unchanged)
# ============================================================
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

    df_renamed = df_pivot.rename(columns={
        "date": "Date",
        "TMAX": "T_High_C",
        "TMIN": "T_Low_C",
        "PRCP": "Precip mm",
        "AWND": "Wind Sp. m/s",
        "RHAV": "Rel Hum %"
    })

    for col in ["T_High_C", "T_Low_C", "Precip mm"]:
        if col in df_renamed.columns:
            df_renamed[col] = df_renamed[col] / 10.0

    df_renamed["Date"] = pd.to_datetime(df_renamed["Date"])
    return df_renamed

# ============================================================
# NOAA 48-hr forecast (unchanged)
# ============================================================
def get_noaa_48hr_forecast(lat, lon):
    """
    Robust hourly forecast fetch from api.weather.gov.
    Weather.gov requires a User-Agent with contact info.
    Returns a DataFrame with at least columns: startTime, shortForecast, PoP.
    """
    try:
        headers = {
            # <-- Put a real contact here per NWS policy
            "User-Agent": "IrrigationDashboard/1.0 (contact: pamori2@huskers.unl.edu)",
            "Accept": "application/geo+json",
        }

        # 1) Discover the forecast endpoints for this lat/lon
        meta_url = f"https://api.weather.gov/points/{lat},{lon}"
        meta_resp = requests.get(meta_url, headers=headers, timeout=12)
        meta_resp.raise_for_status()
        props = meta_resp.json().get("properties", {})
        hourly_url = props.get("forecastHourly")

        if not hourly_url:
            # No hourly URL available for this coordinate → return empty
            return pd.DataFrame()

        # 2) Fetch the hourly forecast
        forecast_resp = requests.get(hourly_url, headers=headers, timeout=12)
        forecast_resp.raise_for_status()
        periods = forecast_resp.json().get("properties", {}).get("periods", [])

        df = pd.DataFrame(periods)
        if df.empty:
            return df

        # Normalize probability of precipitation (PoP) into a simple numeric column
        if "probabilityOfPrecipitation" in df.columns:
            def _pop(v):
                if isinstance(v, dict):
                    return v.get("value", 0) or 0
                return 0
            df["PoP"] = df["probabilityOfPrecipitation"].apply(_pop)
        else:
            df["PoP"] = 0

        return df

    except Exception as e:
        # You can log or surface this if you like:
        # st.caption(f"NOAA forecast error: {e}")
        return pd.DataFrame()


# ============================================================
# Sidebar inputs
# ============================================================
st.sidebar.subheader("📌 Field Information")
field_name = st.sidebar.text_input("Enter Field Name:", value="SCAL Field")
field_lat = st.sidebar.number_input("Latitude (°)", value=40.580, format="%.6f")
field_lon = st.sidebar.number_input("Longitude (°)", value=-98.129, format="%.6f")

st.sidebar.subheader("📥 Load Daily Input for SWD Predictions")

# Local file upload (unchanged behavior)
raw_file = st.sidebar.file_uploader("Upload daily input (CSV):", type=["csv"])

# One-click demo: load same CSV from GitHub
default_url = "https://raw.githubusercontent.com/PreciousAmori/irrigation-dashboard-phase2/main/data/SCAL_Corn_Field_2023.csv"
load_default_btn = st.sidebar.button("📥 Load daily input data from GitHub", key="load_default")


# Trigger predictions
predict_button = st.sidebar.button("🚀 Generate SWD Predictions", key="generate_swd_button")


st.sidebar.header("🔧 Model Parameters")
FC = st.sidebar.number_input("Field Capacity (m³/m³)", value=0.41, step=0.01)
WP = st.sidebar.number_input("Wilting Point (m³/m³)", value=0.19, step=0.01)
f_dc = st.sidebar.slider("Fraction for Maximum Allowable Depletion (MAD fraction)", 0.1, 0.9, 0.5)

st.sidebar.header("📅 Key Dates")
emergence_date = pd.to_datetime(st.sidebar.date_input("🌱 Emergence Date (CGDD start):", pd.to_datetime("2023-05-17")))
date_max = pd.to_datetime(st.sidebar.date_input("📌 Date Max (root depth max / Kcr max):", pd.to_datetime("2023-07-11")))
date_min = emergence_date + pd.Timedelta(days=1)

st.sidebar.subheader("📍 Weather Station Parameters")
elevation = st.sidebar.number_input("Elevation (m)", value=556.0)
latitude_deg = st.sidebar.number_input("Latitude (°)", value=40.57)
longitude_deg = st.sidebar.number_input("Longitude (°)", value=-98.13)
wind_hgt = st.sidebar.number_input("Wind Measurement Height (m)", value=3.0)
albedo = st.sidebar.number_input("Albedo", value=0.23, step=0.01)
Cn = st.sidebar.number_input("Cn", value=1600.0)
Cd = st.sidebar.number_input("Cd", value=0.38)

noaa_stations = {
    "Mead (USC00256325)": {"id": "GHCND:USC00256325", "lat": 41.165, "lon": -96.430},
    "Mead (alt) (USC00256326)": {"id": "GHCND:USC00256326", "lat": 41.172, "lon": -96.478},
    "Clay Center (USC00251610)": {"id": "GHCND:USC00251610", "lat": 40.580, "lon": -98.129},
    "North Platte (USW00024023)": {"id": "GHCND:USW00024023", "lat": 41.088, "lon": -100.763},
    "Brule (USC00250860)": {"id": "GHCND:USC00250860", "lat": 41.029, "lon": -101.971},
    "Scottsbluff (USW00024045)": {"id": "GHCND:USW00024045", "lat": 41.893, "lon": -103.684},
}

st.sidebar.subheader("📡 Weather Source")
use_api = st.sidebar.checkbox("Use Mesonet API", value=True)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-05-03"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-09-30"))


# If the user clicks "Load from GitHub", fetch the CSV and store it for use
if load_default_btn:
    try:
        r = requests.get(default_url, timeout=20)
        r.raise_for_status()
        st.session_state["raw_df_default"] = pd.read_csv(io.StringIO(r.text))
        st.success(f"✅ Loaded {len(st.session_state['raw_df_default']):,} rows from GitHub.")
    except Exception as e:
        st.error(f"Couldn't fetch default CSV: {e}")
        
# Optional 1-liner UX cue
#if raw_file is not None:
#    st.sidebar.success("Using uploaded CSV.")
#elif "raw_df_default" in st.session_state:
#    st.sidebar.info("Using GitHub demo CSV.")


# --- Choose the raw input source for predictions ---
# Decide the source of the raw input: uploaded file takes precedence; otherwise use GitHub-loaded
source_df = None
if raw_file is not None:
    try:
        raw_file.seek(0)                  # ← rewind the buffer
        source_df = pd.read_csv(raw_file)
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV appears to be empty or could not be read.")
        source_df = None
    except Exception as e:
        st.error(f"Couldn't read uploaded CSV: {e}")
elif "raw_df_default" in st.session_state:
    source_df = st.session_state["raw_df_default"]


# Optional tiny debug line:
st.sidebar.caption(f"Demo loaded: {'raw_df_default' in st.session_state}")


# --- Quick, live checklist for users (place AFTER the sidebar inputs, BEFORE Mesonet/NOAA fetch) ---
def _tick(done: bool) -> str:
    return "✅" if done else "⬜"

have_field_info   = bool(field_name)
have_raw_input    = (raw_file is not None) or ("raw_df_default" in st.session_state)
have_predictions  = "pred_df" in st.session_state
have_weather      = "weather_df" in st.session_state
weather_source_chosen = True  # you already present Mesonet/NOAA UI

st.markdown(
    f"""
### 📋 How to use
1. {_tick(have_field_info)} **Enter field info** (name & location)  
2. {_tick(have_raw_input)} **Load daily input data** (upload CSV or use GitHub button)  
3. {_tick(have_predictions)} **Click _🚀 Generate SWD Predictions_**  
4. {_tick(True)} **Enter agronomic info** (FC, WP, MAD fraction, Key dates)  
5. {_tick(True)} **Enter weather station info (Mesonet or NOAA)  
   *(Tip: deselect **Use Mesonet API** to activate NOAA.)* 
6. {_tick(have_weather)} **Click _📡 Fetch Weather Data_**  

> **DEMO:** Use the defaults so everything matches:
> - **Mesonet Station:** *Memphis 5N*  
> - **Daily Input:** GitHub demo CSV (matches Memphis 5N dates/area)  
>
> In production, pick the station that **services your field** and make sure the **daily input** corresponds to the same field & season.
"""
)


# ============================================================
# Mesonet / NOAA fetch (agreport + chunking)  — UI + logic
# ============================================================

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_agreport_station_list(network: str = "nemesonet") -> list[str]:
    """
    Try several public list endpoints and return a unique, sorted list of station names.
    Some Mesonet deployments return an empty list for 'agreport' or require/ignore 'network'.
    We also fall back to the older 'scqc1440' list (names are the same).
    """
    url = "https://awdn.unl.edu/productdata/get"
    names: set[str] = set()

    attempts = [
        {"list": "agreport", "network": network},
        {"list": "agreport"},                     # no network
        {"list": "scqc1440", "network": network}, # fallback: scqc list
        {"list": "scqc1440"},                     # fallback: scqc, no network
    ]

    for params in attempts:
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                names.update(d.get("name") for d in data
                             if isinstance(d, dict) and d.get("name"))
        except Exception:
            # ignore and try the next variant
            pass

    return sorted(n for n in names if n)


if use_api:
    # --- MESONET UI ---
    st.sidebar.subheader("📡 Mesonet (agreport)")
    stations = fetch_agreport_station_list()

    if stations:
        default_idx = stations.index("Memphis 5N") if "Memphis 5N" in stations else 0
        meso_name = st.sidebar.selectbox(
            "Mesonet Station (agreport)",
            stations,
            index=default_idx,
            help="Choose the station name as shown on Mesonet."
        )
    else:
        st.sidebar.info("Couldn't load the Mesonet station list. Type the name manually.")
        meso_name = st.sidebar.text_input(
            "Mesonet Station (agreport)",
            value="Harvard 4SW",
            help="Exact station name, e.g., 'Harvard 4SW'."
        )

    # Optional manual override regardless of dropdown
    manual_station = st.sidebar.text_input("Or override with a station name", "")
    if manual_station.strip():
        meso_name = manual_station.strip()


    tz_choice = st.sidebar.selectbox(
        "Mesonet timezone for day alignment",
        ["EST", "CST6CDT", "UTC", "MST", "PST"],
        index=0,  # EST works reliably with agreport
        help="This only affects which local midnight a day is assigned to."
    )
    chunk_days = st.sidebar.number_input(
        "Chunk size (days per request)", min_value=10, max_value=50, value=30, step=1,
        help="Smaller chunks reduce timeouts on long ranges."
    )
    st.sidebar.text_input("Fallback NOAA Station ID (optional if Mesonet fails)", key="fallback_noaa_id")

    if st.sidebar.button("📡 Fetch Weather Data (Mesonet)"):
        with st.spinner("Fetching Mesonet agreport (chunked, metric)…"):
            try:
                weather_df = get_mesonet_weather(
                    meso_name,
                    start_date,
                    end_date,
                    tz=tz_choice,
                    chunk_days=int(chunk_days),
                    connect_timeout=10,
                    read_timeout=120,
                    verbose=False,
                    show_progress=True,
                )
                if weather_df.empty:
                    raise ValueError("Mesonet returned no rows in the requested window.")
                st.success(f"✅ Mesonet data loaded for '{meso_name}'")
                st.session_state["weather_df"] = weather_df
            except Exception as e:
                st.warning(f"⚠️ Mesonet fetch failed: {e} — trying fallback NOAA...")
                fallback_id = st.session_state.get("fallback_noaa_id", "")
                if not fallback_id:
                    st.error("❌ No fallback NOAA ID provided.")
                else:
                    try:
                        weather_df = fetch_noaa_weather(
                            noaa_token,
                            fallback_id,
                            pd.to_datetime(start_date).strftime("%Y-%m-%d"),
                            pd.to_datetime(end_date).strftime("%Y-%m-%d"),
                        )
                        if weather_df.empty:
                            st.error("❌ NOAA fallback returned no data.")
                        else:
                            st.success(f"✅ NOAA fallback data loaded from {fallback_id}")
                            st.session_state["weather_df"] = weather_df
                    except Exception as ne:
                        st.error(f"❌ NOAA fallback failed: {ne}")

else:
    # --- NOAA UI ---
    st.sidebar.subheader("📡 NOAA Weather Station")
    selected_station = st.sidebar.selectbox(
        "Select a NOAA Station",
        list(noaa_stations.keys()),
        index=None,
        placeholder="Choose a station..."
    )
    fallback_noaa_manual = st.sidebar.text_input(
        "Or manually enter NOAA Station ID (overrides dropdown)", ""
    )

    station_id = None
    if selected_station:
        meta = noaa_stations[selected_station]
        station_id = meta["id"]
        st.sidebar.success(f"Selected: {selected_station}")
        st.sidebar.markdown(f"**Station ID:** {station_id}")
        st.sidebar.markdown(f"**Coordinates:** {meta['lat']:.3f}°, {meta['lon']:.3f}°")

    if st.sidebar.button("📡 Fetch Weather Data (NOAA)"):
        with st.spinner("Fetching NOAA data..."):
            station_to_use = fallback_noaa_manual or station_id
            if not station_to_use:
                st.error("❌ Please select or enter a NOAA station ID.")
            else:
                try:
                    weather_df = fetch_noaa_weather(
                        noaa_token,
                        station_to_use,
                        pd.to_datetime(start_date).strftime("%Y-%m-%d"),
                        pd.to_datetime(end_date).strftime("%Y-%m-%d"),
                    )
                    if weather_df.empty:
                        st.error("❌ NOAA fetch returned no data.")
                    else:
                        st.success(f"✅ NOAA data loaded from {station_to_use}")
                        st.session_state["weather_df"] = weather_df
                except Exception as e:
                    st.error(f"❌ NOAA API error: {e}")


# ============================================================
# File uploaders
# ============================================================
pred_file = st.sidebar.file_uploader("📄 Upload SWD Predictions CSV", type=["csv"])
weather_file = st.sidebar.file_uploader("☀️ Upload Weather CSV", type=["csv"])

# ============================================================
# Manual weather CSV → take precedence if provided
# Recognizes either:
#  (A) your 8-col metric schema [Date, T_High_C, ..., Precip mm]
#  (B) raw Mesonet "agreport" CSV (auto-converts via to_metric_8cols)
# ============================================================
if weather_file is not None:
    try:
        up = pd.read_csv(weather_file)
        up.columns = [c.strip() for c in up.columns]

        metric_cols = {
            "Date", "T_High_C", "T_Low_C", "Rel Hum %", "Soil Tmp C@10cm",
            "Wind Sp. m/s", "SolarRad MJ/m^2/d", "Precip mm"
        }

        if metric_cols.issubset(set(up.columns)):
            # Already metric format
            df_up = up.copy()
            df_up["Date"] = pd.to_datetime(df_up["Date"], errors="coerce").dt.normalize()
        elif "Timestamp" in up.columns and "Max Temperature (F)" in up.columns:
            # Raw Mesonet agreport → convert
            df_up = to_metric_8cols(up)
        else:
            st.error(
                "Unrecognized weather CSV format.\n\n"
                "Expected either the 8-column metric format "
                "or a Mesonet 'agreport' CSV."
            )
            df_up = None

        if df_up is not None:
            df_up = df_up.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            st.session_state["weather_df"] = df_up
            st.success(
                f"✅ Using uploaded weather CSV "
                f"({len(df_up)} rows: {df_up['Date'].min().date()} → {df_up['Date'].max().date()})."
            )
    except Exception as e:
        st.error(f"Couldn't read weather CSV: {e}")


# ============================================================
# Predictions pipeline (upload OR GitHub default; fallback to pred_file)
# ============================================================
source_df = None
if raw_file is not None:
    try:
        raw_file.seek(0)  # <-- critical on Streamlit Cloud
        source_df = pd.read_csv(raw_file)
    except pd.errors.EmptyDataError:
        st.sidebar.error("Uploaded file looks empty. Please re-upload.")
    except Exception as e:
        st.sidebar.error(f"Couldn't read uploaded CSV: {e}")
elif "raw_df_default" in st.session_state:
    source_df = st.session_state["raw_df_default"]


if predict_button:
    if source_df is None:
        st.error("Please upload a CSV or click 'Load daily input data from GitHub' first.")
    else:
        raw_df = source_df.copy()

        # Flexible ID column (2 lines)
        id_col = "Management Plot ID" if "Management Plot ID" in raw_df.columns else "Management_Plot_ID"
        if "Date" not in raw_df.columns or id_col not in raw_df.columns:
            st.error("Input must contain 'Date' and 'Management Plot ID' (or 'Management_Plot_ID').")
        else:
            id_df = raw_df[["Date", id_col]].copy()
            X_features = raw_df.drop(columns=["Date"], errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0.0)

        
            model = joblib.load(model_path)

            # Align to model columns
            exp = getattr(model, "feature_names_in_", None)
            if exp is not None:
                for c in exp:
                    if c not in X_features.columns:
                        X_features[c] = 0.0

                X_features = X_features[exp]

            preds = model.predict(X_features)
         

            predictions_df = id_df.copy()
            predictions_df["SWD_predictions"] = preds
            predictions_df.rename(columns={id_col: "Management_Plot_ID"}, inplace=True)
            predictions_df["Date"] = pd.to_datetime(predictions_df["Date"], errors="coerce").dt.normalize()
            predictions_df = predictions_df.dropna(subset=["Date"]).reset_index(drop=True)

            st.session_state["pred_df"] = predictions_df
            st.success("✅ SWD Predictions generated successfully! Scroll down to proceed.")


# Optional: allow dropping in a ready-made predictions CSV
if "pred_df" not in st.session_state and pred_file is not None:
    try:
        tmp = pd.read_csv(pred_file)
        if "Date" in tmp.columns:
            tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce").dt.normalize()
            tmp = tmp.dropna(subset=["Date"]).reset_index(drop=True)
        st.session_state["pred_df"] = tmp
        st.success("✅ Loaded precomputed predictions CSV.")
    except Exception as e:
        st.error(f"Couldn't read predictions CSV: {e}")

# ============================================================
# Main analysis (needs pred_df & weather_df)
# ============================================================
if 'pred_df' in st.session_state and 'weather_df' in st.session_state:
    pred_df = st.session_state['pred_df'].copy()
    weather_df = st.session_state['weather_df'].copy()

    # --- Clean columns + coerce both Date columns to *true* datetimes (midnight) ---
    weather_df.columns = weather_df.columns.str.strip()

    pred_df['Date']    = pd.to_datetime(pred_df['Date'], errors='coerce').dt.normalize()
    weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors='coerce').dt.normalize()

    # Drop any rows where Date failed to parse (prevents merge crashes)
    pred_df = pred_df.dropna(subset=['Date'])
    weather_df = weather_df.dropna(subset=['Date'])

    # (Optional debug – shows you the dtypes in the UI)
    st.caption(f"Date dtypes → pred_df: {pred_df['Date'].dtype} | weather_df: {weather_df['Date'].dtype}")


    # === GDD & CGDD ===
    Tbase = 10.0
    Tmax_cap = 30.0
    weather_df['Tmax_Lim'] = weather_df['T_High_C'].clip(lower=Tbase, upper=Tmax_cap)
    weather_df['Tmin_Lim'] = weather_df['T_Low_C'].clip(lower=Tbase, upper=Tmax_cap)
    weather_df['Tavg'] = (weather_df['T_High_C'] + weather_df['T_Low_C']) / 2
    weather_df['GDD'] = ((weather_df['Tmax_Lim'] + weather_df['Tmin_Lim']) / 2) - Tbase
    weather_df['GDD'] = weather_df['GDD'].clip(lower=0)

    weather_df = weather_df.sort_values('Date')
    weather_df['CGDD'] = 0.0
    mask = weather_df['Date'] >= emergence_date
    weather_df.loc[mask, 'CGDD'] = weather_df.loc[mask, 'GDD'].cumsum()

    # === ETr ===
    phi = math.pi * latitude_deg / 180
    Gsc = 4.92
    sigma = 4.901e-9
    G = 0.0
    weather_df['doy'] = weather_df['Date'].dt.dayofyear
    weather_df['es'] = (
        0.6108 * np.exp((17.27 * weather_df['T_High_C'])/(weather_df['T_High_C']+237.3)) +
        0.6108 * np.exp((17.27 * weather_df['T_Low_C'])/(weather_df['T_Low_C']+237.3))
    ) / 2
    weather_df['ea'] = (weather_df['Rel Hum %'] / 100.0) * weather_df['es']
    weather_df['dr'] = 1 + 0.033 * np.cos(2 * np.pi * weather_df['doy'] / 365)
    weather_df['delta'] = 0.409 * np.sin((2 * np.pi * weather_df['doy'] / 365) - 1.39)
    weather_df['ws'] = np.arccos(-np.tan(phi) * np.tan(weather_df['delta']))
    weather_df['Ra'] = (24/np.pi * Gsc * weather_df['dr'] *
                        (weather_df['ws']*np.sin(phi)*np.sin(weather_df['delta']) +
                         np.cos(phi)*np.cos(weather_df['delta'])*np.sin(weather_df['ws'])))

    if 'SolarRad MJ/m^2/d' not in weather_df.columns:
        kRs = 0.19 if latitude_deg < 20 else 0.16
        st.info(f"☀️ Estimating solar radiation using Hargreaves method with kRs={kRs}")
        weather_df['SolarRad MJ/m^2/d'] = (
            kRs * np.sqrt((weather_df['T_High_C'] - weather_df['T_Low_C']).clip(lower=0)) * weather_df['Ra']
        )
        weather_df['SolarRad MJ/m^2/d_source'] = 'Estimated (Hargreaves)'
    else:
        weather_df['SolarRad MJ/m^2/d_source'] = 'Observed'

    weather_df['Rso'] = (0.75 + 2e-5 * elevation) * weather_df['Ra']
    weather_df['fcd'] = 1.35 * np.minimum(np.maximum(weather_df['SolarRad MJ/m^2/d'] / weather_df['Rso'], 0.3), 1.0) - 0.35
    weather_df['Rns'] = (1 - albedo) * weather_df['SolarRad MJ/m^2/d']
    weather_df['Rnl'] = sigma * weather_df['fcd'] * (0.34 - 0.14*np.sqrt(np.maximum(weather_df['ea'], 0))) * (
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
    df = pred_df.merge(
    weather_df[['Date', 'GDD', 'CGDD', 'ETr']],
    on='Date',
    how='left'
    )

    # === Kcr / ETa
    def compute_kcr(cgdd):
        if cgdd < 100: return 0.4
        elif cgdd < 600: return 0.4 + 0.0012*(cgdd-100)
        elif cgdd < 1400: return 1.0
        elif cgdd < 1700: return 1.0 + (-0.0027)*(cgdd-1400)
        else: return 0.2
    df['Kcr'] = df['CGDD'].apply(compute_kcr)
    df['ETa'] = df['ETr'] * df['Kcr']

    # === Water balance
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

    # Optional forecast suggestion
    if irrigation_recommended:
        st.warning("⚠️ Irrigation is recommended. Consider checking upcoming rainfall forecasts.")
        if st.checkbox("🔍 View 48-hour NOAA Rainfall Forecast", key="forecast_toggle"):
            forecast_df = get_noaa_48hr_forecast(field_lat, field_lon)
            if forecast_df.empty:
                st.error("❌ Forecast could not be loaded. Please check internet connection or NOAA API.")
            else:
                forecast_df["time"] = pd.to_datetime(forecast_df["startTime"])
                forecast_df["Hour"] = range(len(forecast_df))
                now_utc = datetime.now(timezone.utc)
                forecast_start = forecast_df["time"].iloc[0]
                current_hour_index = int((now_utc - forecast_start).total_seconds() // 3600)

                line = alt.Chart(forecast_df).mark_line(point=True).encode(
                    x=alt.X("Hour:Q", title="Forecast Hour (0–155)"),
                    y=alt.Y("PoP:Q", title="Precipitation Probability (%)"),
                    tooltip=["time:T", "shortForecast:N", "PoP:Q"]
                ).properties(title="🌧️ 48-Hour NOAA Rainfall Forecast")

                now_rule = alt.Chart(pd.DataFrame({"Hour": [current_hour_index]})).mark_rule(
                    color="red", strokeDash=[5, 5], strokeWidth=3
                ).encode(x="Hour:Q")

                st.altair_chart((line + now_rule).interactive(), use_container_width=True)
                forecast_df["Hour"] = range(len(forecast_df))
                st.dataframe(
                    forecast_df[["Hour", "shortForecast", "PoP"]].head(10).rename(
                        columns={"shortForecast": "Forecast", "PoP": "Precipitation Probability (%)"}
                    )
                )

    # === Dashboard
    plot_ids = df['Management_Plot_ID'].unique()
    selected_id = st.selectbox("Select a Plot ID:", plot_ids)
    filtered = df[df['Management_Plot_ID'] == selected_id]

    min_date = filtered['Date'].min().date()
    max_date = filtered['Date'].max().date()
    date_range = st.slider("📅 Select Date Range:", min_value=min_date, max_value=max_date,
                           value=(min_date, max_date), format="MM/DD/YYYY")
    filtered = filtered[(filtered['Date'].dt.date >= date_range[0]) & (filtered['Date'].dt.date <= date_range[1])]

    metrics = st.multiselect(
        "📊 Select variables to plot:",
        ["SWD_predictions", "MAD", "TD", "Kcr", "ETr", "ETa"],
        default=["SWD_predictions"]
    )

    melted = filtered.melt(id_vars=['Date'], value_vars=metrics, var_name='Variable', value_name='Value')
    color_map = {'SWD_predictions': 'blue','MAD': 'red','TD': 'green','Kcr': 'purple','ETr': 'gray','ETa': 'brown'}

    line_chart = alt.Chart(melted).mark_line(strokeWidth=2).encode(
        x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%m/%d', labelAngle=-45, labelFontSize=12, titleFontSize=14)),
        y=alt.Y('Value:Q', title='Values (mm or unit)', axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
        color=alt.Color('Variable:N',
                        scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                        legend=alt.Legend(title="Variables", labelFontSize=12, titleFontSize=14, orient='top', direction='horizontal')),
        tooltip=['Date:T', 'Variable:N', 'Value:Q']
    ).properties(width=1500, height=450)

    merged_df = pd.merge(
        filtered[['Date', 'Recommended_Irrigation_mm']],
        weather_df[['Date', 'Precip mm']],
        on='Date', how='inner'
    )
    plot_df = merged_df.melt(id_vars='Date', var_name='Metric', value_name='Value')

    bars = alt.Chart(plot_df[plot_df['Metric'] == 'Recommended_Irrigation_mm']).mark_bar(opacity=0.7).encode(
        x=alt.X('Date:T', axis=alt.Axis(format='%m/%d', labelAngle=-45)),
        y=alt.Y('Value:Q', title='Depth of water (mm)'),
        color=alt.value('orange'),
        tooltip=['Date:T', 'Metric:N', 'Value:Q']
    )
    lines = alt.Chart(plot_df[plot_df['Metric'] == 'Precip mm']).mark_line(strokeWidth=2, color='#2B9CE2').encode(
        x='Date:T', y='Value:Q', tooltip=['Date:T', 'Metric:N', 'Value:Q']
    )
    combined_chart = alt.layer(bars, lines).properties(
        width=1500, height=400, title='Irrigation Recommendations with Precipitation Overlay'
    )

    st.altair_chart(line_chart, use_container_width=True)
    st.altair_chart(combined_chart, use_container_width=True)

    st.write("### 📑 Recommendations Table")
    cols_to_move = ["GDD", "CGDD", "ETr", "Kcr", "ETa", "Recommended_Irrigation_mm"]
    other_cols = [c for c in filtered.columns if c not in cols_to_move]
    ordered = filtered[other_cols + cols_to_move]
    st.dataframe(ordered.round(1))

    st.download_button(
        "📥 Download CSV",
        data=ordered.to_csv(index=False).encode('utf-8'),
        file_name=f"Irrigation_Recommendations_{field_name.replace(' ','_')}.csv",
        mime='text/csv'
    )
