import pandas as pd

rawCrimes = pd.read_csv("Crimes.csv")
rawPoverty = pd.read_csv("Poverty.csv")
rawUnemployment = pd.read_csv("unemployment.csv")
rawArea = pd.read_csv("area.csv")

print(rawPoverty)

crimesPoverty = pd.merge(rawCrimes, rawPoverty,  how='left', left_on=['year','state_name'], right_on = ['year','state'])
crimesPoverty.drop(columns = ["5"])

print(crimesPoverty)