import fastf1
import pandas as pd
from collections import Counter

# Enable FastF1 cache and load race session
fastf1.Cache.enable_cache('f1_cache')
year, gp, session = 2024, 'Belgium Grand Prix', 'R'
race = fastf1.get_session(year, gp, session)
race.load()

# Get all laps and filter only accurate non-out-laps
laps = race.laps
race_laps = laps[(laps['LapNumber'] > 1) & (laps['IsAccurate'] == True)]

# Determine the most used compound and filter by it
compound_counts = Counter(race_laps['Compound'])
most_common_compound = compound_counts.most_common(1)[0][0]
race_laps = race_laps[race_laps['Compound'] == most_common_compound]

# Filter clean air laps
race_laps = race_laps[race_laps['Time'].notnull() & race_laps['Position'].notnull()].sort_values(by='LapStartTime')

clean_air_laps = []
for lap in race_laps.itertuples():
    ahead = race_laps[(race_laps['LapStartTime'] == lap.LapStartTime) & (race_laps['Position'] == lap.Position - 1)]
    if ahead.empty:
        clean_air_laps.append(lap)

clean_air_df = pd.DataFrame(clean_air_laps)

# Manually set track length for Belgium 
track_length_km = 7.004

# Group by driver and calculate pace
data = []
for driver in clean_air_df['Driver'].unique():
    drv_laps = clean_air_df[clean_air_df['Driver'] == driver]
    avg_pace = drv_laps['LapTime'].mean().total_seconds()
    norm_pace = avg_pace / track_length_km if track_length_km else None
    data.append({'Driver': driver, 'Average Race Pace': avg_pace, 'Normalized Race Pace (s/km)': norm_pace})

# Create DataFrame, sort, and display
df = pd.DataFrame(data).sort_values(by='Normalized Race Pace (s/km)')

print(f"\nUsing compound: {most_common_compound}")
print(f"Track length used for normalization: {track_length_km:.3f} km")
print(f"\nFound {len(clean_air_df)} clean air laps after filtering.\n")
print(df.to_string(index=False))