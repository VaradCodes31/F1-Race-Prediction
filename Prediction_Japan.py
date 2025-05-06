import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer  # Importing Imputer

# Enable the FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load the data for the Japanese GP race session, lap, and sector times
session_24 = fastf1.get_session(2024, "Japan", "R")
session_24.load()
laps_2024 = session_24.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert Lap Times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# Manually input qualifying data for 2025 Japanese GP
quali_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [86.983, 86.995, 87.027, 87.299, 87.318, 87.610, 87.822, 87.897, 88.000, 87.386, 88.570, 88.696, 89.271]
})

# Wet performance factor
driver_wet_performance = {
    "VER": -2.480371,
    "HAM": -2.353646,
    "LEC": -2.413849,
    "NOR": -2.182075,
    "ALO": -2.734495,
    "RUS": -3.132181,
    "SAI": -2.124629,
    "TSU": -0.366205,
    "OCO": -1.819039,
    "GAS": -2.116771,
    "STR": -2.014251
}

quali_2025["WetPerformanceFactor"] = quali_2025["Driver"].map(driver_wet_performance)

# Weather Data
API_KEY = "f2ec0f95e60e03ddb5d25c9b86cac2e7"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?q=Tokyo&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()

# Extracting relevant weather details (2 PM Local Time)
forecast_time = "2025-04-05 14:00:00"
forecast_data = None

# Adjusted to check the forecast list correctly
if "list" in weather_data:
    for forecast in weather_data['list']:
        if forecast["dt_txt"] == forecast_time:
            forecast_data = forecast
            break

if forecast_data:
    rain_probability = forecast_data["pop"] * 100  # Converting to percentage
    temperature = forecast_data["main"]["temp"]
else:
    rain_probability = 0
    temperature = 20

# Merging Qualifying data with sector times data
merged_data = quali_2025.merge(sector_times_2024, on="Driver", how="left")

# Creating weather features for the model
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Use the average lap time per driver as the target
avg_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
merged_data = merged_data.merge(avg_lap_times, on="Driver", how="left")
y = merged_data["LapTime (s)"]

# Defining feature set
X = merged_data[[
    "QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "WetPerformanceFactor", "Temperature", "RainProbability"
]]

# ✅ Handling Missing Values: Using Mean Imputation
imputer = SimpleImputer(strategy='mean')  # Impute missing values with the mean
X_imputed = imputer.fit_transform(X)

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict the race times using 2025 qualifying and sector information
predicted_race_times = model.predict(X_imputed)
quali_2025["PredictedRaceTime (s)"] = predicted_race_times
quali_2025 = quali_2025.sort_values("PredictedRaceTime (s)")

print("\n 🏁 Winner of the 2025 Japanese GP: 🏁\n")
print(quali_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate
y_pred = model.predict(X_test)
print(f"\n🔎 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
