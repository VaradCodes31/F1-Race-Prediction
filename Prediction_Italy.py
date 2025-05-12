import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error 
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("f1_cache")

# Load the data for the Miami GP race session
session_24 = fastf1.get_session(2024, "Miami", "R")
session_24.load()
laps_2024 = session_24.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# 2025 Miami GP Qualifying Data
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR"],
    "QualifyingTime (s)": [86.204, 86.269, 86.375, 86.754, 86.385, 87.006, 87.710, 87.604, 86.943, 86.569, 87.473, 86.824, 87.830]
})

# Wet performance factors of drivers
driver_wet_performance = {
    "VER": 2.641581, "HAM": 2.956423, "LEC": 1.718659, "NOR": 0.781900,
    "ALO": 3.830736, "RUS": 3.651065, "SAI": 3.125732, "TSU": 2.884163,
    "OCO": 3.258957, "GAS": 3.404705, "STR": 3.232278, "PIA": 1.334407
}
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

# Get weather forecast
API_KEY = "f2ec0f95e60e03ddb5d25c9b86cac2e7"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?q=Tokyo&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()
forecast_time = "2025-05-05 16:00:00"
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else 20

# Apply wet factor if rain probability is high
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] + qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Team performance scores
team_points = {
    "McLaren": 246, "Mercedes": 141, "Red Bull": 105, "Williams": 37, "Ferrari": 94,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 8, "Alpine": 7
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

# Race pace data
race_pace_data = {
    "LEC": 17.188549, "VER": 17.152932, "PIA": 16.998759, "HAM": 17.225083,
    "ANT": 17.160323, "RUS": 17.162470, "ALB": 17.164001, "SAI": 17.187282,
    "NOR": 17.057782, "LAW": 17.430904, "ALO": 17.319281, "BEA": 17.295020,
    "HAD": 17.243451, "DOO": 15.480023, "OCO": 17.247822, "STR": 17.296368,
    "HUL": 17.276173, "BOR": 17.248368, "TSU": 17.203481, "GAS": 17.283768,
}
qualifying_2025["RacePace"] = qualifying_2025["Driver"].map(race_pace_data)

# Merge with 2024 lap times
avg_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
merged_data = qualifying_2025.merge(avg_lap_times, on="Driver", how="left")

# Impute missing lap times with mean
mean_lap_time = merged_data["LapTime (s)"].mean()
merged_data["LapTime (s)"] = merged_data["LapTime (s)"].fillna(mean_lap_time)

# Also impute any remaining NaNs in input features
merged_data["RacePace"] = merged_data["RacePace"].fillna(merged_data["RacePace"].mean())
merged_data["TeamPerformanceScore"] = merged_data["TeamPerformanceScore"].fillna(merged_data["TeamPerformanceScore"].mean())

# Add weather
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Features and target
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "RacePace"
]]
y = merged_data["LapTime (s)"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=38)
model.fit(X_train, y_train)

# Predict
merged_data["PredictedLapTime (s)"] = model.predict(X)
final_results = merged_data.sort_values("PredictedLapTime (s)")

# Show winner
print("Winner of the 2025 Miami GP: ")
print(final_results[["Driver", "PredictedLapTime (s)"]])

# Model error
y_pred = model.predict(X_test)
print(f"\n🔎 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Plot: Team Performance vs Predicted Time
plt.figure(figsize=(12, 8))
plt.scatter(final_results["TeamPerformanceScore"],
            final_results["PredictedLapTime (s)"],
            c=final_results["QualifyingTime"])

for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["TeamPerformanceScore"].iloc[i],
                          final_results["PredictedLapTime (s)"].iloc[i]), fontsize=9,
                 xytext=[5, 5], textcoords="offset points")

plt.colorbar(label="Qualifying Time")
plt.xlabel("Team Performance Score")
plt.ylabel("Predicted Race Time")
plt.title("Effect of Team Performance Score on Predicted Race Results")
plt.tight_layout()
plt.savefig("team_performance_effect.png")
plt.show()

# Plot feature importance
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Prediction Model")
plt.tight_layout()
plt.show()
