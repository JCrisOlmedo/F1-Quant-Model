import fastf1
import pandas as pd

fastf1.Cache.enable_cache("cache_folder")

# Load the 2023 Canadian GP (wet race)
wetsession2023 = fastf1.get_session(2023,"Canada","R")
wetsession2023.load()

# Load the 2022 Canadian GP (dry race)
drysession2022 = fastf1.get_session(2022,"Canada","R")
drysession2022.load()

# Extract the laps from both races:
wetlaps = wetsession2023.laps[["Driver","LapTime"]].copy()
drylaps = drysession2022.laps[["Driver","LapTime"]].copy()

# Clean data
wetlaps.dropna(inplace=True)
drylaps.dropna(inplace=True)

# Convert to seconds
wetlaps["LapTime (s)"] = wetlaps["LapTime"].dt.total_seconds()
drylaps["LapTime (s)"] = drylaps["LapTime"].dt.total_seconds()

# Calculate the average lap time for each driver in both races:
avg_wet_lap = wetlaps.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_dry_lap = drylaps.groupby("Driver")["LapTime (s)"].mean().reset_index()

# Merge
merged_data = pd.merge(avg_dry_lap,avg_wet_lap,on="Driver",suffixes=[" Dry"," Wet"])

# Calculate the performance difference in lap times between wet and dry:
merged_data["LapTimeDiff (s)"] = merged_data["LapTime (s) Wet"] - merged_data["LapTime (s) Dry"]

merged_data["PerformanceChange (%)"] = (merged_data["LapTimeDiff (s)"] / merged_data["LapTime (s) Dry"]) * 100

# Performance Score
merged_data["WetPerformanceScore"] = 1 + (merged_data["PerformanceChange (%)"] / 100)

# Print out the wet performance scores for each drive:
print("\n🌧️ Driver Wet Performance Scores (2023 vs. 2022) 🌧️: ")
print(merged_data[["Driver","WetPerformanceScore"]].sort_values(by="WetPerformanceScore"))
merged_data[["Driver","WetPerformanceScore"]].to_csv("Wet Performance Score.csv")