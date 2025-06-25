import fastf1
import pandas as pd
import numpy as np
from data_pipeline import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


"""
    2025 Australian F1 GP 
"""

# Enable FastF1 caching
fastf1.Cache.enable_cache("cache_folder")

# 2024 Race session data.
session_2024 = fastf1.get_session(2024,'Canada','R')
session_2024.load()

# -- Extract lap times --
# We .copy() cause we are modifying the data.
laps_2024 = session_2024.laps[["Driver","LapTime"]].copy()
print(laps_2024)


# -- Clean the data --
# Subset to select where we are looking for NAs
# Inplace = True to modify the current df rather than creating a new one.
laps_2024.dropna(subset=["LapTime"],inplace=True)
# Create a new column to have the time in seconds:
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()


# 2025 Qualifying session data:
qf_2025 = pipeline("https://www.formula1.com/en/results/2025/races/1263/canada/qualifying")


# -- Merge the data --

# Left_on to tell what column are we joining the data on the left df
# Right_on to tell what column are we joinign the data on the right df.
merged_data = qf_2025.merge(laps_2024, left_on="Driver", right_on="Driver")
# Use only "QualifyingTime (s)" as a feature for our model. This is because we want to estimate the lap times using the qualifying times:
X = merged_data[["QualifyingTime (s)"]]
y = merged_data[["LapTime (s)"]]
if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources.")


# -- Train Gradient Boosting Model --

# Data splitting.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ML model generation.
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)

# Model fitting:
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qf_2025[["QualifyingTime (s)"]])
qf_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qf_2025 = qf_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏆 Predicted 2025 Australian GP Winner 🏆\n")
print(qf_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n📉 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")