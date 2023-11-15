import pandas as pd

rawCrimes = pd.read_csv("Crimes.csv")
rawPoverty = pd.read_csv("Poverty.csv")
rawUnemployment = pd.read_csv("unemployment.csv")
rawArea = pd.read_csv("area.csv")

crimesPoverty = pd.merge(rawCrimes, rawPoverty,  how='left', left_on=['year','state_name'], right_on = ['year','state'])
crimesPoverty = crimesPoverty[crimesPoverty['state'].notna()] # cleans out all previous years in the crimes dataset

cPU = pd.merge(crimesPoverty, rawUnemployment,  how='left', left_on=['year','state_name'], right_on = ['year','state'])

print(cPU)