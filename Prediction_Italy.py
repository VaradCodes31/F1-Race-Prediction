import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error 
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("f1_cache")

# Load the data for the Saudi Arabian GP race session, lap, and sector times
session_24 = fastf1.get_session(2024, "Saudi Arabia", "R")
session_24.load()
laps_2024 = session_24.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector times by driver 
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

# Calculate total sector time
sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] + 
    sector_times_2024["Sector2Time (s)"] + 
    sector_times_2024["Sector3Time (s)"]
)

# 2025 Saudi Arabian GP Qualifying Data
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [87.294, 87.481, 87.304, 87.670, 87.407, 88.201, 88.367, 88.303, 88.204, 88.164, 88.782, 89.092, 88.645]
})

# Average Lap Times for 2025 season of drivers
avg_lap_times_2025 = {
    "VER": 88.0, "PIA": 89.1, "LEC": 89.2, "RUS": 89.3, "HAM": 89.4,
    "GAS": 89.5, "ALO": 89.6, "TSU": 89.7, "SAI": 89.8, "HUL": 89.9,
    "OCO": 90.0, "STR": 90.1, "NOR": 90.2
}

# Wet performance factors of drivers
driver_wet_performance = {
    "VER": 1.408441,
    "HAM": 0.725168,
    "LEC": 3.338737,
    "NOR": 4.605577,
    "ALO": 0.294132,
    "RUS": 0.429713,
    "SAI": 0.978754,
    "TSU": 0.492332,
    "OCO": 0.159084,
    "GAS": -19.696676,
    "STR": 0.208399
}

qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

API_KEY = "f2ec0f95e60e03ddb5d25c9b86cac2e7"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?q=Tokyo&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()
forecast_time = "2025-04-20 18:00:00"
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else 20

# Wet performance factor only to be considered if rain probability is above 75%
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] + qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Add the Constructors points for better calculation
team_points = {
    "McLaren": 78, "Mercedes": 53, "Red Bull": 36, "Williams": 17, "Ferrari": 17,
    "Haas": 14, "Aston Martin": 10, "Kick Sauber": 8, "Racing Bulls": 3, "Alpine": 0
}

max_points = max(team_points.values())
team_performance_score = {team: (points / max_points) for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari",
    "RUS": "Mercedes", "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin",
    "TSU": "Racing Bulls", "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine",
    "STR": "Aston Martin"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

avg_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()

merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data = merged_data.merge(avg_lap_times, on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Feature definition
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore"
]]

y = merged_data["LapTime (s)"]
clean_data = merged_data.dropna()

X = clean_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore"
]]
y = clean_data["LapTime (s)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=38)
model.fit(X_train, y_train)
clean_data["PredictedLapTime (s)"] = model.predict(X)

final_results = clean_data.sort_values("PredictedLapTime (s)")
print("2025 Saudi Arabian GP Winner: ")
print(final_results[["Driver", "PredictedLapTime (s)"]])

# MAE
y_pred = model.predict(X_test)
print(f"\n🔎 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Plot effects of team performance score
plt.figure(figsize=(12, 8))
plt.scatter(final_results["TeamPerformanceScore"],
            final_results["PredictedLapTime (s)"],
            c=final_results["QualifyingTime"])

for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["TeamPerformanceScore"].iloc[i], final_results["PredictedLapTime (s)"].iloc[i]), fontsize=9,
                 xytext = [5,5], textcoords="offset points")

plt.colorbar(label="Qualifying Time")
plt.xlabel("Team Performance Score")
plt.ylabel("Predicted Race Time")
plt.title("Effect of Team Performance Score on predicted Race Results")
plt.tight_layout()
plt.savefig("team_performance_effect.png")
plt.show()

# Plot how important each feature is in the model
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Prediction Model")
plt.tight_layout()
plt.show()
