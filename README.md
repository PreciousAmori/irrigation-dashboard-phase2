# 🌱 Irrigation Recommendation Dashboard – Phase 2

This Streamlit application provides field-scale irrigation recommendations using a transferable machine learning framework developed from multi-environment field data across Nebraska. The dashboard integrates weather data, agronomic parameters, and a Leave-One-Site-Out (LOSO) XGBoost model to predict Soil Water Depletion (SWD) and support irrigation decision-making across diverse production environments.

**Author:** Precious Amori

---

## Overview

The Phase 2 dashboard is powered by a transferable XGBoost model trained using data from multiple field sites spanning Nebraska's hydroclimatic gradient. The framework was developed to improve model generalization beyond a single field or season and to support irrigation recommendations under diverse environmental conditions.

---

## Features

* Transferable LOSO XGBoost SWD prediction model
* Nebraska Mesonet (agreport) weather integration
* NOAA weather data and forecast support
* Automatic calculation of:

  * Growing Degree Days (GDD)
  * Cumulative GDD (CGDD)
  * Reference Evapotranspiration (ETr)
  * Crop Evapotranspiration (ETa)
* Plot-level irrigation recommendations
* Weather data upload support
* Interactive visualizations and downloadable recommendations

---

## Model Information

**Model:**

```text
XGB_loso_scal.joblib
```

**Training Strategy:**

* Multi-site training
* Leave-One-Site-Out (LOSO) validation
* Developed using field datasets spanning Nebraska's east-west hydroclimatic gradient

---

## Quick Start

```bash
git clone https://github.com/PreciousAmori/irrigation-dashboard-phase2.git

cd irrigation-dashboard-phase2

pip install -r requirements.txt

streamlit run app.py
```

---

## Demo Dataset

The repository includes a demonstration dataset:

```text
data/SCAL_Corn_Field_2023.csv
```

To use:

1. Load the demo dataset from GitHub or upload manually.
2. Generate SWD predictions.
3. Enter agronomic parameters.
4. Fetch weather data.
5. Review irrigation recommendations and download results.

---

## Repository Structure

```text
irrigation-dashboard-phase2/
├── app.py
├── data/
│   └── SCAL_Corn_Field_2023.csv
├── models/
│   └── trained/
│       └── XGB_loso_scal.joblib
├── requirements.txt
└── README.md
```

---

## Typical Workflow

1. Enter field information.
2. Load daily input data.
3. Generate SWD predictions.
4. Enter agronomic parameters.
5. Fetch weather data.
6. Review irrigation recommendations.
7. Download recommendation results.

---

## Deployment

### Streamlit Community Cloud

1. Push repository to GitHub.
2. Create a new Streamlit application.
3. Connect the repository.
4. Add required secrets (NOAA token if used).
5. Deploy.

---

## Acknowledgements

This project was developed as part of research on precision irrigation and transferable machine learning for agricultural water management.

Data sources and supporting organizations include:

* Nebraska Mesonet (UNL)
* NOAA
* USDA-NIFA Cyber-Physical Systems (CPS) Project
* University of Nebraska–Lincoln
* USDA NRCS Web Soil Survey
* Planet Labs PBC

```
```
