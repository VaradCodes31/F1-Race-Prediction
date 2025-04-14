# 🏎️ F1 2025: Race Prediction Model

This repository contains machine learning-based **race time predictions** for **all races in the Formula 1 2025 season**. The model combines qualifying results, historical sector times, weather forecasts, and wet-weather performance metrics to predict each driver’s average race lap time for a given Grand Prix.

The project is built entirely in **Python** and will be updated **throughout the season**, race by race.

---

## 🎯 Project Objective

To build and maintain a predictive machine learning model for every Grand Prix in the **2025 Formula 1 season**, using:
- Sector-wise race performance data from previous GPs
- Official qualifying times
- Live weather forecasts
- Custom driver-specific **Wet Performance Factor**

---

## 📦 Tech Stack & Libraries

- **Python 3.9+**
- `FastF1` – to retrieve F1 timing and telemetry data
- `pandas`, `numpy` – for data manipulation
- `scikit-learn` – for machine learning models (Gradient Boosting Regressor)
- `requests` – to pull weather forecasts via API
- `OpenWeatherMap` – for real-time weather data

---

## 🗃️ Data Sources

- **FastF1 API**: Race and sector data from the 2024 season and beyond.
- **Manual Entry**: 2025 qualifying times (until APIs are updated).
- **OpenWeatherMap API**: Forecasts for each Grand Prix location.
- **Custom Script**: Wet Performance Factor based on historical wet race performance. *(See `wet_performance_index.py`)*

---

## ⚙️ Features Used in Model

| Feature                | Description |
|------------------------|-------------|
| `QualifyingTime (s)`   | Driver's qualifying lap time in seconds |
| `Sector1Time (s)`      | Average sector 1 race time from previous GP |
| `Sector2Time (s)`      | Average sector 2 race time from previous GP |
| `Sector3Time (s)`      | Average sector 3 race time from previous GP |
| `WetPerformanceFactor` | Driver's performance adjustment in wet conditions |
| `Temperature`          | Forecast temperature (°C) |
| `RainProbability`      | Rain chance (%) at race time |

---

## 🧠 Model Overview

- **Algorithm**: Gradient Boosting Regressor (from Scikit-learn)
- **Preprocessing**: Mean imputation for missing data
- **Train/Test Split**: 80/20
- **Evaluation Metric**: Mean Absolute Error (MAE)

---

## 🏁 Current Race: Bahrain GP 2025

```text
🏆 Predicted Winner: VER
🔢 Example Output:

Driver    | Predicted Race Time (s)
--------  | ------------------------
VER       | 96.561
SAI       | 96.843 
...       | ...
```

> 🔍 **Model MAE**: ~`X.XX` seconds (based on backtesting with 2024 data)

---

## 🌧️ Wet Performance Factor

A separate Python script, [`wet_performance_index.py`](wet_performance_index.py), is used to compute a custom **WetPerformanceFactor** for each driver. This is based on their average lap delta in historically wet races.

---

## 📁 Project Structure

```
📦 f1-2025-race-prediction/
 ┣ 📄 Prediction_Japan.py          ← [Race-specific script]
 ┣ 📄 Prediction_Bahrain.py        ← [Race-specific script]
 ┣ 📄 wet_performance_index.py     ← [Driver wet index calculator]
 ┣ 📄 README.md                     
 ┣ 📁 f1_cache/                    ← [FastF1 cache files]
 ┗ 📄 requirements.txt
```

---

## 🚀 How to Run

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/f1-2025-race-prediction.git
cd f1-2025-race-prediction
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run prediction for a race**:
```bash
python japan_gp_prediction.py
```

---

## 📈 Upcoming Updates

- ✅ Scripts for future races will follow the same pattern
- ✅ Visualizations: driver deltas, model confidence intervals
- ✅ Dashboard (optional) for quick race/weekend overview

---

## 🙌 Acknowledgments

- [FastF1](https://theoehrly.github.io/Fast-F1/)
- [OpenWeatherMap](https://openweathermap.org/)
- [Scikit-learn](https://scikit-learn.org/)

---
