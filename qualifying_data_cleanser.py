import sys
import pandas as pd




def qf_cleanser(race,data):
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
    df = df.reset_index(drop=True)  # Reset index without adding it as a column
    df.index = df.index + 1  # Adjust index to start from 1


    # Clean drivers names
    df['Driver'] = df['Driver'].str.split(" ", n=1).str[1]

    # Time processing to total seconds:

    for i in range(2,11):
        df["Time"][i] = df["Time"][i].split("+")[1]

    df["Minutes"] = df["Time"].str.split(":").str[0]
    df["Seconds"] = df["Time"].str.split(":").str[1]

    df["QualifyingTime (s)"] = df["Seconds"]
    df["QualifyingTime (s)"][1] = float(df["Minutes"][1])*60+float(df["Seconds"][1])

    for i in range(2,11):
        df["QualifyingTime (s)"][i] = df["QualifyingTime (s)"][1]+float(df["Time"][i])

    for i in range(11, 21):
        df["QualifyingTime (s)"][i] = float(df["Minutes"][i])*60+float(df["Seconds"][i])

    race = f"/Quant Model for F1/QualifyingData{race}.csv"
    qualy_data =df[["Driver","QualifyingTime (s)"]]
    qualy_data.to_csv(race)

if __name__ == "__main__":
    # ------ INPUT ------
    # Read multiline input
    race = input("What's this race?\n")

    print("Paste your lines (Ctrl+D to end):")
    data = sys.stdin.read()
    
    qf_cleanser(race,data)