import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

# Load Bahrain 2024 race session data
session_24 = fastf1.get_session(2024, "Bahrain", "R")
session_24.load()
laps_2024 = session_24.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# Manually input qualifying times for 2025 Japanese GP (reused structure for prediction purposes)
quali_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [90.423, 90.267, 89.841, 90.175, 90.009, 90.772, 90.216, 91.886, 91.303, 90.680, 92.067, 91.594, 92.283]
})

# Fixed weather conditions for Bahrain GP (Sunny Day)
rain_probability = 0  # no rain
temperature = 27      # approximate sunny day temperature in Bahrain in Celsius

# Merge qualifying and sector data
merged_data = quali_2025.merge(sector_times_2024, on="Driver", how="left")

# Add constant weather conditions
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Define target as average lap time per driver
avg_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
merged_data = merged_data.merge(avg_lap_times, on="Driver", how="left")
y = merged_data["LapTime (s)"]

# Define feature set (WetPerformanceFactor removed)
X = merged_data[[
    "QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "Temperature", "RainProbability"
]]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict race times
predicted_race_times = model.predict(X_imputed)
quali_2025["PredictedRaceTime (s)"] = predicted_race_times
quali_2025 = quali_2025.sort_values("PredictedRaceTime (s)")

# Output prediction
print("\n 🏁 2025 Bahrain GP Winner: 🏁\n")
print(quali_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate model
y_pred = model.predict(X_test)
print(f"\n🔎 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
