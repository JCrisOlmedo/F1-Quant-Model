import sys
import pandas as pd

def qf_cleanser(race,data):
    # # Split into lines
    # lines = data.strip().split('\n')

    # # Split each line into parts
    # rows = [line.split('\t') for line in lines]

    # # ------ DataFrame ------
    # df = pd.DataFrame(rows, columns=["Driver","Team","Time"])
    df = data

    # Indexation
    df = df.reset_index(drop=True)  # Reset index without adding it as a column

    # Time processing to total seconds:
    
    df["Q1 Minutes"] = df["Q1"].str.split(":").str[0].astype(float)
    df["Q1 Seconds"] = df["Q1"].str.split(":").str[1].astype(float)

    df["Q1 (s)"] = df["Q1 Minutes"] * 60 + df["Q1 Seconds"]
    
    df["Q2 Minutes"] = df["Q2"].str.split(":").str[0].astype(float)
    df["Q2 Seconds"] = df["Q2"].str.split(":").str[1].astype(float)

    df["Q2 (s)"] = (df["Q2 Minutes"]) * 60 + (df["Q2 Seconds"])
    
    df["Q3 Minutes"] = df["Q3"].str.split(":").str[0].astype(float)
    df["Q3 Seconds"] = df["Q3"].str.split(":").str[1].astype(float)

    df["Q3 (s)"] = (df["Q3 Minutes"]) * 60 + (df["Q3 Seconds"])
 

    df["QualifyingTime (s)"] = df[["Q1 (s)","Q2 (s)","Q3 (s)"]].min(axis=1)



    driver_mapping = {"Alexander Albon": "ALB",   "Fernando Alonso": "ALO",       "Kimi Antonelli": "ANT",
                    "Oliver Bearman": "BEA",    "Gabriel Bortoleto": "BOR",     "Valtteri Bottas": "BOT",
                    "Franco Colapinto": "COL",  "Nyck de Vries": "DEV",         "Jack Doohan": "DOO",
                    "Pierre Gasly": "GAS",      "Isack Hadjar": "HAD",          "Lewis Hamilton": "HAM",
                    "Nico Hulkenberg": "HUL",   "Liam Lawson": "LAW",           "Charles Leclerc": "LEC",
                    "Kevin Magnussen": "MAG",   "Lando Norris": "NOR",          "Esteban Ocon": "OCO",
                    "Sergio Pérez": "PER",      "Oscar Piastri": "PIA",         "Daniel Ricciardo": "RIC",
                    "George Russell": "RUS",    "Carlos Sainz": "SAI",          "Logan Sargeant": "SAR",
                    "Lance Stroll": "STR",      "Yuki Tsunoda": "TSU",          "Max Verstappen": "VER",
                    "Zhou Guanyu": "ZHO"}
    df["Driver"] = df["Driver"].str.strip()
    qualy_data =df[["Driver","QualifyingTime (s)"]]
    qualy_data["DriverCode"] = qualy_data["Driver"].map(driver_mapping)
    print(qualy_data)
    qualy_data.reset_index(drop=True)
    qualy_data.to_csv(race+"QualifyingData.csv")
    return qualy_data

if __name__ == "__main__":

    raw_data = pd.read_csv("JapanQualyRawData.txt",sep="\t")

    qf_cleanser("Japan",raw_data)