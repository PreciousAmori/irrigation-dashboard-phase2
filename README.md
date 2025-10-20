# 🌱 Irrigation Recommendation Dashboard

Streamlit app that combines **Mesonet (agreport)** and **NOAA** weather with an XGBoost model to predict **Soil Water Depletion (SWD)** and generate field-level irrigation recommendations.  
Includes CGDD, ETr/ETa, precipitation overlays, a demo dataset loader, and manual weather CSV upload.

---

## ✨ Features

- Mesonet **agreport** fetch (chunked, robust, metric conversion)
- NOAA daily data + 48-hr forecast
- Manual **weather CSV** upload (8-column metric or raw agreport)
- One-click **demo daily input** loader from GitHub
- SWD prediction (XGBoost) + irrigation recommendation table & charts
- Clear in-app checklist + demo defaults (Memphis 5N)

---

## 🚀 Quickstart (Local)

```bash
# 1) clone
git clone https://github.com/PreciousAmori/irrigation-dashboard.git
cd irrigation-dashboard

# 2) create env (examples)
# conda create -n irrigation-ml python=3.10 -y && conda activate irrigation-ml
# or: python -m venv .venv && . .venv/Scripts/activate  # (Windows)

# 3) install
pip install -r requirements.txt

# 4) secrets for local runs
# create: .streamlit/secrets.toml  (see below)

# 5) run
streamlit run app.py



🧪 Demo mode

In the sidebar:

Click “📥 Load daily input data from GitHub”

Keep Mesonet Station = “Memphis 5N”

Click “🚀 Generate SWD Predictions”

Enter agronomic params if needed

Fetch weather via Mesonet (or uncheck to use NOAA)

Demo CSV lives at
data/ImplementationSET_corn_complete.csv


Project structure

irrigation-dashboard/
├─ app.py
├─ data/
│  └─ ImplementationSET_corn_complete.csv
├─ models/
│  └─ trained/
│     ├─ XGBoost_vs4.pkl
│     └─ scaler_vs4.pkl
├─ requirements.txt
└─ .streamlit/
   └─ secrets.toml           # (local only)


📄 CSV Schemas
A) Daily input (for predictions)

Must include at least:

Date (any parseable date format; normalized internally)

Management Plot ID (or Management_Plot_ID)

All other columns are model features and are scaled automatically.


B) Weather (metric 8-column schema)

You can upload this directly, or it’s produced from Mesonet agreport:

| Column            | Units     |
| ----------------- | --------- |
| Date              | —         |
| T_High_C          | °C        |
| T_Low_C           | °C        |
| Rel Hum %         | %         |
| Soil Tmp C@10cm   | °C        |
| Wind Sp. m/s      | m/s       |
| SolarRad MJ/m^2/d | MJ/m²/day |
| Precip mm         | mm        |


C) Raw Mesonet agreport (auto-converted)

If you upload the CSV directly from Mesonet agreport, the app converts it to the 8-column metric table. It expects headers like:

Timestamp

Max Temperature (F), Min Temperature (F)

Relative Humidity (%)

Soil Temperature at 4 inches (F)

Solar Radiation (MJ/m^2/day)

Precipitation (in)

Wind Speed (mph)


🧭 Typical workflow

Enter field info → name & coordinates

Load daily input (upload CSV or GitHub button)

Click 🚀 Generate SWD Predictions

Set agronomic parameters (FC, WP, MAD)

Choose weather source

Keep Use Mesonet API for Mesonet agreport

Uncheck to switch to NOAA

📡 Fetch Weather Data

Explore charts & download the recommendations CSV


🧰 Troubleshooting

Model not found
Ensure models/trained/XGBoost_vs4.pkl and models/trained/scaler_vs4.pkl exist (case-sensitive paths on Linux).

Manual daily CSV upload fails
The app now uses the file-like buffer directly and rewinds before reading. If issues persist, make sure the file is a valid CSV and not empty.

NOAA forecast missing
NOAA requires a User-Agent with contact info; this is set in the code. Ensure NOAA_TOKEN is configured and your internet connection is healthy.


📦 Deployment (Streamlit Cloud)

Push this repo to GitHub

In Streamlit Cloud: New app → Connect repo → Select branch

Add Secrets (NOAA_TOKEN)

Deploy


🤝 Contributing

PRs welcome! Please open an issue first if you’d like to propose larger changes.


📜 License

MIT © 2025 Precious Amori


🙏 Acknowledgements

This project integrates multiple open datasets and APIs to support research in precision irrigation and nitrogen management.

Special thanks to:

- **Nebraska Mesonet (UNL)** — for providing weather and soil data via the `agreport` API  
- **NOAA NCEI / Weather.gov** — for daily and hourly meteorological datasets  
- **Planet Labs PBC** — for providing access to *PlanetScope* satellite imagery used in related analysis and validation workflows  
- **USDA NRCS Web Soil Survey (WSS)** — for soil property and classification data supporting model input and field interpretation  
- **USDA-NIFA Cyber-Physical Systems (CPS) Project** — for funding and research support  
- **University of Nebraska–Lincoln, Department of Biological Systems Engineering** — for field operations, instrumentation, and academic support  

> *PlanetScope imagery © 2023–2025 Planet Labs PBC, used under research license.*  
> *Soil data © USDA NRCS Web Soil Survey (WSS).*