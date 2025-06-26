import fastf1
import pandas as pd
import numpy as np
from data_pipeline import pipeline,points
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


"""
    2025 Bahrain F1 GP 
"""

fastf1.Cache.enable_cache("cache_folder")

# 2024 Race session data.
session_2024 = fastf1.get_session(2024,'Bahrain','R')
session_2024.load()
laps_2024 = session_2024.laps[["Driver","LapTime","Sector1Time","Sector2Time","Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime","Sector1Time","Sector2Time","Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)","Sector2Time (s)","Sector3Time (s)"]].mean().reset_index()

qf_2025 = pipeline("https://www.formula1.com/en/results/2025/races/1263/bahrain/qualifying")

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

season_points = points("https://web.archive.org/web/20250408064919/https://www.formula1.com/en/results/2025/drivers")

# qf_2025["SeasonPoints"] = qf_2025.merge(season_points, left_on="Driver", )

# print(qf_2025)

# Weather data:
API_KEY = "ce3f2cdf97820fc02c0436cb99fe3f78"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=34.8823&lon=136.5845&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()

forecast_time = "2025-04-30 15:00:00"
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else 20


merged_data = qf_2025.merge(sector_times_2024, left_on="Driver", right_on="Driver", how='inner')
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

X = merged_data[["QualifyingTime (s)", 
                 "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", 
                 "WetPerformanceFactor", 
                 "RainProbability", 
                 "Temperature"]].fillna(1)

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources.")

y = merged_data.merge(laps_2024.groupby("Driver")["LapTime (s)"].mean(), left_on="Driver",right_index=True)["LapTime (s)"]

# -- Train Gradient Boosting Model --
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)
model.fit(X_train, y_train)

predicted_lap_times = model.predict(X)
merged_data["PredictedRaceTime (s)"] = predicted_lap_times
merged_data["PredictedRaceTime"] = pd.to_datetime(merged_data["PredictedRaceTime (s)"], unit='s').apply(lambda x: x.strftime("%M:%S.%f"))

# Rank drivers by predicted race time
merged_data = merged_data.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏆 Predicted 2025 Bahrain GP Winner 🏆\n")
print(merged_data[["Driver", "PredictedRaceTime (s)", "PredictedRaceTime"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n📉 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")