import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

fastf1.Cache.enable_cache("f1_cache")

# Load the 2024 Belgian GP session data (previous race for Barcelona)
session_2024 = fastf1.get_session(2024, 'Belgium Grand Prix', "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector times by driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# Clean air race pace from Belgian GP 2024 (using your racepace.py on Spanish GP data)
clean_air_race_pace = {
    "VER": 108.393, "HAM": 107.539, "LEC": 107.707, "NOR": 107.390, "ALO": 109.102,
    "PIA": 107.426, "RUS": 108.001, "SAI": 108.561, "STR": 109.328, "HUL": 110.211,
    "OCO": 108.558, "GAS": 109.037, "ALB": 108.605
}

# Updated qualifying data for Belgian GP 2025
qualifying_2025 = pd.DataFrame({
    "Driver": ["PIA", "NOR", "VER", "RUS", "HAM", "LEC", "SAI", "ALB", 
               "OCO", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime (s)": [  
        71.546,  # PIA (1:11.546)
        71.755,  # NOR (1:11.755)
        71.848,  # VER (1:11.848)
        71.848,  # RUS (1:11.848)
        72.045,  # HAM (1:12.045)
        72.131,  # LEC (1:12.131)
        73.203,  # SAI (1:13.203)
        72.641,  # ALB (1:12.641)
        73.201,  # OCO (1:13.201)
        73.058,  # STR (1:13.058)
        72.199,  # GAS (1:12.199)
        72.284,  # ALO (1:12.284)
        73.190   # HUL (1:13.190)
    ]
})
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# Get weather data for Wallonia, Belgium
API_KEY = "f2ec0f95e60e03ddb5d25c9b86cac2e7"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=41.5697&lon=2.2581&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()
forecast_time = "2025-07-26 13:30:00"  # Belgium GP race time
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else 25

# Adjust qualifying time based on weather conditions
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * 1.05  # 5% penalty for wet
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# Add constructor's data (2025 season standings)
team_points = {
    "McLaren": 460, "Ferrari": 222, "Mercedes": 210, "Red Bull": 172, "Williams": 59,
    "Haas": 29, "Aston Martin": 36, "Kick Sauber": 41, "Racing Bulls": 36, "Alpine": 19
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin", "ALB": "Williams"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# Average position change at Belgium GP (qualifying pos - finish pos); negative means losing positions
average_position_change_belgium = {
    "VER": -3.0,  # 3 places Lost
    "NOR": 0.0,   # Steady
    "PIA": 4.0,   # 4 places gained
    "RUS": -7.0,  # DQ'd in the race
    "SAI": 2.0,   # 2 places gained
    "ALB": -1.0,  # 1 place lost
    "LEC": -1.0,  # 1 place lost
    "OCO": 1.0,   # 1 place gained
    "HAM": 3.0,   # 3 places gained
    "STR": 4.0,   # 4 places gained
    "GAS": -1.0,  # 1 place lost
    "ALO": 1.0,   # 1 place gained
    "HUL": -2.0   # 2 places lost
}
qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_belgium)

# Merge qualifying and sector times data
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data["QualifyingTime"] = merged_data["QualifyingTime"]

valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers]

# Define features (X) and target (y)
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", 
    "CleanAirRacePace (s)", "AveragePositionChange"
]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# Impute missing values for features
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)

# Train XGBoost model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=37,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train, y_train)
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# Sort the results to find the predicted winner
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Belgian GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# Plot effect of clean air race pace
plt.figure(figsize=(12, 8))
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"])
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')
plt.xlabel("Clean Air Race Pace (s)")
plt.ylabel("Predicted Race Time (s)")
plt.title("Effect of Clean Air Race Pace on Predicted Belgian GP Results")
plt.tight_layout()
plt.show()

# Plot feature importances
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Belgian GP Race Time Prediction")
plt.tight_layout()
plt.show()

# Sort results and get top 3
final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]

print("\n🏆 Predicted Belgian GP Top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}")
print(f"🥉 P3: {podium.iloc[2]['Driver']}")