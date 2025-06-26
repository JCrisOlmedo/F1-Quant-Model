import fastf1
import pandas as pd
import numpy as np
from data_pipeline import pipeline
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


"""
    2025 Japan F1 GP 
"""

# Enable FastF1 caching
fastf1.Cache.enable_cache("cache_folder")

# 2024 Race session data.
session_2024 = fastf1.get_session(2024,'Japan','R')
session_2024.load()

# -- Extract lap times --
# We .copy() cause we are modifying the data.
laps_2024 = session_2024.laps[["Driver","LapTime","Sector1Time","Sector2Time","Sector3Time"]].copy()


# -- Clean the data --
# Inplace = True to modify the current df rather than creating a new one.
laps_2024.dropna(inplace=True)

# Create a new column to have the time in seconds:
for col in ["LapTime","Sector1Time","Sector2Time","Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get the average sector times per driver:
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)","Sector2Time (s)","Sector3Time (s)"]].mean().reset_index()

# 2025 Qualifying session data:
qf_2025 = pipeline("https://www.formula1.com/en/results/2025/races/1263/japan/qualifying")


# -- Wet Performance Factor -- 
wet_performance = {
    "ALB": 0.978120,
    "ALO": 0.972655,
    "BOT": 0.982052,
    "GAS": 0.978832,
    "HAM": 0.976464,
    "LEC": 0.975862,
    "MAG": 0.989983,
    "NOR": 0.978179,
    "OCO": 0.981810,
    "PER": 0.998904,
    "RUS": 0.968678,
    "SAI": 0.978754,
    "STR": 0.979857,
    "TSU": 0.996338,
    "VER": 0.975196,
    "ZHO": 0.987774,
}

qf_2025["WetPerformanceFactor"] = qf_2025['Driver'].map(wet_performance, na_action='ignore')


# Weather data:
API_KEY = "ce3f2cdf97820fc02c0436cb99fe3f78"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=34.8823&lon=136.5845&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()

# Extract the relevant weather data for the race (Sunday at 2pm local time)
forecast_time = "2025-04-05 14:00:00"
forecast_data = None
for forecast in weather_data["list"]:
    if forecast["dt_txt"] == forecast_time:
        forecast_data = forecast
        break

if forecast_data:
    rain_probability = forecast_data["pop"]
    temperature = forecast_data["main"]["temp"]  
else:
    rain_probability = 0 
    temperature = 20 

# -- Merge the data --
# Left_on to tell what column are we joining the data on the left df
# Right_on to tell what column are we joinign the data on the right df.
# How = 'left' beacuse we want our qf data untouched.
merged_data = qf_2025.merge(sector_times_2024, left_on="Driver", right_on="Driver", how='inner')

# Weather features:
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

# Feature set: (QualifyingTime (s) + SectorTimes (s) + Weather + Wet Performance Factor)
X = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "WetPerformanceFactor", "RainProbability", "Temperature"]].fillna(1)

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources.")

# Use average lap time as the target. Merged data didn't contained the LapTime (s)
y = merged_data.merge(laps_2024.groupby("Driver")["LapTime (s)"].mean(), left_on="Driver",right_index=True)["LapTime (s)"]

# -- Train Gradient Boosting Model --

# Data splitting.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ML model generation.
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)

# Model fitting:
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(X)
merged_data["PredictedRaceTime (s)"] = predicted_lap_times
merged_data["PredictedRaceTime"] = pd.to_datetime(merged_data["PredictedRaceTime (s)"], unit='s').apply(lambda x: x.strftime("%M:%S.%f"))

# Rank drivers by predicted race time
merged_data = merged_data.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏆 Predicted 2025 Japanese GP Winner 🏆\n")
print(merged_data[["Driver", "PredictedRaceTime (s)", "PredictedRaceTime"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n📉 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")