import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from data_pipeline import pipeline

"""
    2025 Chinese F1 GP 
"""

# Enable FastF1 caching
fastf1.Cache.enable_cache("cache_folder")

# 2024 Race session data.
session_2024 = fastf1.get_session(2024,'China','R')
session_2024.load()

# -- Extract lap times --
# We .copy() cause we are modifying the data.
laps_2024 = session_2024.laps[["Driver","LapTime","Sector1Time","Sector2Time","Sector3Time"]].copy()

# -- Clean the data --
# Subset to select where we are looking for NAs
# Inplace = True to modify the current df rather than creating a new one.
laps_2024.dropna(inplace=True)

# Create a columns to have the times in seconds:
for col in ["LapTime","Sector1Time","Sector2Time","Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get the average of the sector time and laptime per driver:
sector_times_2024 = laps_2024.groupby("Driver")[["LapTime (s)","Sector1Time (s)","Sector2Time (s)","Sector3Time (s)"]].mean().reset_index()


# -- 2025 Qualifying session data --
raw_data = """1) Oscar Piastri	McLaren	1:30.641
2) George Russell	Mercedes	+0.082
3) Lando Norris	McLaren	+0.152
4) Max Verstappen	Red Bull	+0.176
5) Lewis Hamilton	Ferrari	+0.286
6) Charles Leclerc	Ferrari	+0.380
7) Isack Hadjar	Racing Bulls	+0.438
8) Kimi Antonelli	Mercedes	+0.462
9) Yuki Tsunoda	Racing Bulls	+0.997
10) Alex Albon	Williams	+1.065
Knocked out in Q2
11) Esteban Ocón	Haas	1:31.625
12) Nico Hulkenberg	Sauber	1:31.632
13) Fernando Alonso	Aston Martin	1:31.688
14) Lance Stroll	Aston Martin	1:31.773
15) Carlos Sainz	Williams	1:31.840
Knocked out in Q1
16) Pierre Gasly	Alpine	1:31.992
17) Oliver Bearman	Haas	1:32.018
18) Jack Doohan	Alpine	1:32.092
19) Gabriel Bortoleto	Sauber	1:32.141
20) Liam Lawson	Red Bull	1:32.174"""

# qualifying_2025 = qf_cleanser("China",raw_data)
qualifying_2025 = pd.read_csv("ChinaQualifyingData.csv",index_col=0)

# -- MERGE --

# Merge sector times with qualifying data
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode",right_on="Driver", how="inner",suffixes=(None,"_drop"))
merged_data.drop(columns="Driver_drop")
merged_data.dropna(how="all")

# Update features to include sector times
X = merged_data[["QualifyingTime (s)","Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]

# Target remains the same
y = merged_data[["LapTime (s)"]]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources.")

# -- Train Gradient Boosting Model --
# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ML model generation
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)

# Model fitting
model.fit(X_train, y_train)

# Predict using 2025 qualifying times and sector times
predicted_lap_times = model.predict(X)
merged_data["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
merged_data = merged_data.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏆 Predicted 2025 Chinese GP Winner 🏆\n")
print(merged_data[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n📉 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")