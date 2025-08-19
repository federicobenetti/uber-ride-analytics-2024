# Uber Ride Analytics 2024 🚕📊

Clean, explore, and visualize the **Uber Ride Analytics Dataset (2024)**.  
This repo includes:
- A **processing pipeline** (`src/process_data.py`) to clean and engineer features
- A **Streamlit dashboard** (`app.py`) for quick exploration
- A **reproducible structure** with `requirements.txt` and `.gitignore`

> Dataset source: Kaggle — *Uber Ride Analytics Dataset 2024*. Please place the CSV inside `data/` (or upload in the Streamlit app).

---

## 📁 Repo Structure
```
uber-ride-analytics-2024/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/                 # put raw CSV here (excluded from git by default)
├── notebooks/
├── results/              # exported charts/tables
└── src/
    └── process_data.py   # cleaning + feature engineering
```

---

## 🚀 Quickstart

### 1) Create & activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Process the CSV (optional – the Streamlit app can also process on upload)
```bash
python src/process_data.py --input data/uber_ride_analytics_2024.csv --output results/uber_2024_processed.parquet
```

### 4) Run the dashboard
```bash
streamlit run app.py
```

Then open the local URL Streamlit shows.

---

## ✨ Features engineered
- **datetime**: parsed from `Date` + `Time`
- **year, month, day, weekday, hour, weekofyear**
- **is_weekend**, **part_of_day** (night/morning/afternoon/evening)
- **is_peak_hour** (7–9 & 17–19 local)
- **trip_completed** (from `Booking Status`)
- **cancellation_party** and unified **cancellation_reason**
- **is_cancelled**, **is_incomplete**
- **fare_per_km**, **high_value_ride** (top 10% by fare)
- **rating_gap** (driver - customer)
- **distance_bin_km** (0–2, 2–5, 5–10, 10+)
- **payment_category** (normalized from multiple methods)
- **vehicle_category** (grouping by vehicle type)

---

## 🧪 Notes
- Column names are normalized to **snake_case**.
- The pipeline is **idempotent** and tolerant to missing columns.
- Outputs can be saved to **Parquet** (recommended) or **CSV**.

---

## 📝 License
N/A
