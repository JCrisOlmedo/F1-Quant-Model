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
    df['Driver'] = df['Driver'].str.replace(r'^\s*\d+\)\s*', '', regex=True)

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


    driver_mapping = {  "Alexander Albon": "ALB",   "Fernando Alonso": "ALO",       "Kimi Antonelli": "ANT",
                    "Oliver Bearman": "BEA",    "Gabriel Bortoleto": "BOR",     "Valtteri Bottas": "BOT",
                    "Franco Colapinto": "COL",  "Nyck de Vries": "DEV",         "Jack Doohan": "DOO",
                    "Pierre Gasly": "GAS",      "Isack Hadjar": "HAD",          "Lewis Hamilton": "HAM",
                    "Nico Hülkenberg": "HUL",   "Liam Lawson": "LAW",           "Charles Leclerc": "LEC",
                    "Kevin Magnussen": "MAG",   "Lando Norris": "NOR",          "Esteban Ocon": "OCO",
                    "Sergio Pérez": "PER",      "Oscar Piastri": "PIA",         "Daniel Ricciardo": "RIC",
                    "George Russell": "RUS",    "Carlos Sainz": "SAI",          "Logan Sargeant": "SAR",
                    "Lance Stroll": "STR",      "Yuki Tsunoda": "TSU",          "Max Verstappen": "VER",
                    "Zhou Guanyu": "ZHO"
}
    
    df["DriverCode"] = df["Driver"].map(driver_mapping)
    qualy_data =df[["DriverCode","Driver","QualifyingTime (s)"]]
    qualy_data.reset_index(drop=True)
    qualy_data.to_csv(race+"QualifyingData.csv")
    return qualy_data

if __name__ == "__main__":

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

    qf_cleanser("China",raw_data)