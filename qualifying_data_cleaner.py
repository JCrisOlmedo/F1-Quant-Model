import sys
import pandas as pd

# ------ INPUT ------
# Read multiline input
print("Paste your lines (Ctrl+D to end):")
data = sys.stdin.read()

# Split into lines
lines = data.strip().split('\n')

# Split each line into parts
rows = [line.split('\t') for line in lines]


# ------ DataFrame ------
df = pd.DataFrame(rows, columns=["Driver","Team","Time"])

# Replace empty strings with NaN and drop rows containing NaN
df.replace("", float("nan"), inplace=True)  # Replace "" with NaN
df.dropna(inplace=True)  # Drop rows with NaN

# Indexation
df_reset = df.reset_index(drop=True)  # Reset index without adding it as a column
df_reset.index = df_reset.index + 1  # Adjust index to start from 1


# Clean drivers names
df['Driver'] = df['Driver'].str.split(" ", n=1).str[1]

# Time processing to seconds.
# Function to convert time to seconds
def time_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

df["Minutes"] = df["Time"].str.split(":").str[0]
df["Seconds"] = df["Time"].str.split(":").str[1]

df["Time (s)"] = float(df["Minutes"])*60+float(df["Seconds"])
print(df)


