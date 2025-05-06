import fastf1
import pandas as pd

# Enable caching
fastf1.Cache.enable_cache("f1_cache")

# Load the 2023 Canadian Grand Prix data (wet race)
session_23 = fastf1.get_session(2023, "Canada", "R")
session_23.load()

# Load the 2022 Canadian Grand Prix Data (dry race)
session_22 = fastf1.get_session(2022, "Canada", "R")
session_22.load()

# Extract the lap times for both races
laps_2023 = session_23.laps[["Driver", "LapTime"]].copy()
laps_2022 = session_22.laps[["Driver", "LapTime"]].copy()

# Drop NaN values
laps_2023.dropna(inplace=True)
laps_2022.dropna(inplace=True)

# Convert Lap Times to seconds
laps_2023["LapTime (s)"] = laps_2023["LapTime"].dt.total_seconds()
laps_2022["LapTime (s)"] = laps_2022["LapTime"].dt.total_seconds()

# Calculate the average lap time for each driver in both races
avg_laps_2023 = laps_2023.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_laps_2022 = laps_2022.groupby("Driver")["LapTime (s)"].mean().reset_index()

# Merge data from both races in Driver column
merged_data = pd.merge(avg_laps_2023, avg_laps_2022, on="Driver", suffixes=("_2023", "_2022"))

# Calculate the performance difference between 2023 and 2022
merged_data["PerformanceChange (%)"] = (merged_data["LapTime (s)_2023"] - merged_data["LapTime (s)_2022"]) / merged_data["LapTime (s)_2022"] * 100

# ✅ Added calculation for LapTimeDiff (s)
merged_data["LapTimeDiff (s)"] = merged_data["LapTime (s)_2023"] - merged_data["LapTime (s)_2022"]

# ✅ Fixed the WetPerformanceChange calculation
merged_data["WetPerformanceChange (%)"] = merged_data["LapTimeDiff (s)"] / merged_data["LapTime (s)_2022"] * 100

# Now, we create the wet performance score for each driver
print("\n Driver Wet Performance Score (2023 vs 2022): ")
print(merged_data[["Driver", "WetPerformanceChange (%)"]])
