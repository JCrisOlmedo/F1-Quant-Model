import pandas as pd
import numpy as np

def pipeline(dir):
    # We want this function to open the .csv and to generate a df that's usable for the predictor.
    og = pd.read_html(dir)[0]
    og = og.head(20)


    # The format that F1Fast uses for pilot names is their code, let's chage that:
    clean = pd.DataFrame(columns=['Driver','QualifyingTime (s)'])
    clean['Driver'] = og['DRIVER'].str.slice(start=-3)

    # Now the times. We need will be modelling with the fastest time of each driver, and its pit postition.

    og['Q1 (m)'] = og['Q1'].str.split(":").str[0].astype(float)
    og['Q1 (secs)'] = og['Q1'].str.split(":").str[1].astype(float)

    clean['Q1'] = og['Q1 (m)'] * 60 + og['Q1 (secs)']

    og['Q2 (m)'] = og['Q2'].str.split(":").str[0].astype(float)
    og['Q2 (secs)'] = og['Q2'].str.split(":").str[1].astype(float)

    clean['Q2'] = og['Q2 (m)'] * 60 + og['Q2 (secs)']
    clean['Q2'].replace(np.nan,1000000,inplace=True)

    og['Q3 (m)'] = og['Q3'].str.split(":").str[0].astype(float)
    og['Q3 (secs)'] = og['Q3'].str.split(":").str[1].astype(float)

    clean['Q3'] = og['Q3 (m)'] * 60 + og['Q3 (secs)']
    clean['Q3'].replace(np.nan,1000000,inplace=True)


    clean["QualifyingTime (s)"] = clean[['Q1','Q2','Q3']].min(axis=1)

    clean = clean[['Driver','QualifyingTime (s)']]

    return clean

    # driver_mapping = {"Alexander Albon": "ALB",   "Fernando Alonso": "ALO",       "Kimi Antonelli": "ANT",
    #                 "Oliver Bearman": "BEA",    "Gabriel Bortoleto": "BOR",     "Valtteri Bottas": "BOT",
    #                 "Franco Colapinto": "COL",  "Nyck de Vries": "DEV",         "Jack Doohan": "DOO",
    #                 "Pierre Gasly": "GAS",      "Isack Hadjar": "HAD",          "Lewis Hamilton": "HAM",
    #                 "Nico Hulkenberg": "HUL",   "Liam Lawson": "LAW",           "Charles Leclerc": "LEC",
    #                 "Kevin Magnussen": "MAG",   "Lando Norris": "NOR",          "Esteban Ocon": "OCO",
    #                 "Sergio Pérez": "PER",      "Oscar Piastri": "PIA",         "Daniel Ricciardo": "RIC",
    #                 "George Russell": "RUS",    "Carlos Sainz": "SAI",          "Logan Sargeant": "SAR",
    #                 "Lance Stroll": "STR",      "Yuki Tsunoda": "TSU",          "Max Verstappen": "VER",
    #                 "Zhou Guanyu": "ZHO"}

def points(url):
    og = pd.read_html(url)[1]
    print(og)

    clean = pd.DataFrame(columns=['Driver','Points'])
    # clean['Driver'] = og['DRIVER'].str.slice(start=-3)
    clean['Driver'] = og['Driver'].str.slice(start=-3)
    # clean['Points'] = og["PTS."]
    clean['SeasonPoints'] = og["Pts"]
    
    return clean
