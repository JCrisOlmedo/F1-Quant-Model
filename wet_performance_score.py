import fastf1
import pandas as pd

fastf1.Cache.enable_cache("cache_folder")

# Load the 2023 Canadian GP (wet race)
session2023 = fastf1.get_session(2023,"Canada","R")
session2023.load()

# Load the 2022 Canadian GP (dry race)
session2022 = fastf1.get_session(2022,"Canada","R")
session2022.load()

# Extract the laps from both races:
laps_2023 = session2023.laps[["Driver","LapTime"]].copy()
laps_2022 = session2022.laps[["Driver","LapTime"]].copy()

# Clean data
laps_2023.dropna(inplace=True)
laps_2022.dropna(inplace=True)

# Convert to seconds
laps_2023["LapTime (s)"] = laps_2023["LapTime"].dt.total_seconds()
laps_2022["LapTime (s)"] = laps_2022["LapTime"].dt.total_seconds()

# Calculate the average lap time for each driver in both races:
avg_lap_2023 = laps_2023.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_lap_2022 = laps_2022.groupby("Driver")["LapTime (s)"].mean().reset_index()

# Merge
merged_data = pd.merge(avg_lap_2023,avg_lap_2022,on="Driver",suffixes=("_2023","_2022"))

# Calculate the performance difference in lap times between wet and dry:
merged_data["LapTimeDiff (s)"] = merged_data["LapTime (s)_2023"] - merged_data["LapTime (s)_2022"]

merged_data["PerformanceChange (%)"] = (merged_data["LapTimeDiff (s)"] / merged_data["LapTime (s)_2022"]) * 100

# Performance Score
merged_data["WetPerformanceScore"] = 1 + (merged_data["PerformanceChange (%)"] / 100)

# Print out the wet performance scores for each drive:
print("\n🌧️ Driver Wet Performance Scores (2023 vs. 2022) 🌧️: ")
print(merged_data[["Driver","WetPerformanceScore"]])
