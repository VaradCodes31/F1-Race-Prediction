import fastf1
from fastf1.core import Laps
import pandas as pd
from collections import Counter

# Enable FastF1 cache
fastf1.Cache.enable_cache('f1_cache')

# Load race session
year = 2025
gp = 'Miami Grand Prix'
session = 'R'

race = fastf1.get_session(year, gp, session)
race.load()

# Get all laps and filter only accurate non-out-laps
laps = race.laps
race_laps = laps[(laps['LapNumber'] > 1) & (laps['IsAccurate'] == True)]

# Determine the most used compound
compound_counts = Counter(race_laps['Compound'])
most_common_compound = compound_counts.most_common(1)[0][0]

# Filter laps by most common compound
race_laps = race_laps[race_laps['Compound'] == most_common_compound]

# Filter clean air laps
race_laps = race_laps[race_laps['Time'].notnull() & race_laps['Position'].notnull()]
race_laps = race_laps.sort_values(by='LapStartTime')

clean_air_laps = []
for lap in race_laps.itertuples():
    ahead = race_laps[
        (race_laps['LapStartTime'] == lap.LapStartTime) & 
        (race_laps['Position'] == lap.Position - 1)
    ]
    if ahead.empty:
        clean_air_laps.append(lap)

# Convert to DataFrame
clean_air_df = pd.DataFrame(clean_air_laps)

# ✅ Manually set track length for Miami
track_length_km = 5.412  # Miami International Autodrome 

# Group by driver and calculate pace
drivers = clean_air_df['Driver'].unique()
data = []

for driver in drivers:
    drv_laps = clean_air_df[clean_air_df['Driver'] == driver]
    avg_pace = drv_laps['LapTime'].mean().total_seconds()
    norm_pace = avg_pace / track_length_km if track_length_km else None

    data.append({
        'Driver': driver,
        'Average Race Pace': avg_pace,
        'Normalized Race Pace (s/km)': norm_pace
    })

# Create DataFrame, sort, and display
df = pd.DataFrame(data).sort_values(by='Normalized Race Pace (s/km)')

print(f"\nUsing compound: {most_common_compound}")
print(f"Track length used for normalization: {track_length_km:.3f} km")
print(f"\nFound {len(clean_air_df)} clean air laps after filtering.\n")
print(df.to_string(index=False))
