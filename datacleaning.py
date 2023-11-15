import pandas as pd

rawCrimes = pd.read_csv("Crimes.csv")
rawPoverty = pd.read_csv("Poverty.csv")
rawUnemployment = pd.read_csv("unemployment.csv")
rawArea = pd.read_csv("area.csv")

rawArea['year'] = 2010

rawAreaRepeat=rawArea.copy()
for i in range(1,14):
    rawArea['year']=2009+i
    rawAreaRepeat=pd.concat([rawAreaRepeat,rawArea])

rawPoverty.drop(["population"], axis = 1, inplace = True)
rawCrimes.drop(["rape_legacy"], axis = 1, inplace = True)

rawCrimes.rename(columns={'state_name': 'state'}, inplace=True)

rawCrimes['crime_total']= rawCrimes.iloc[:, -9:-1].sum(axis=1)
rawCrimes['crime_rate']= 100000 * rawCrimes['crime_total']/rawCrimes['population']
rawCrimes.drop(["violent_crime",
          'homicide',
          'rape_revised',
          'robbery',
          'aggravated_assault',
          'property_crime',
          'burglary',
          'larceny',
          'motor_vehicle_theft',
          'crime_total'], axis=1, inplace=True)

crimesPoverty = pd.merge(rawCrimes, rawPoverty,  how='left', left_on=['year','state'], right_on = ['year','state'])
crimesPoverty = crimesPoverty[crimesPoverty['poverty_percent'].notna()] # cleans out all previous years in the crimes and poverty dataset

cPU = pd.merge(crimesPoverty, rawUnemployment,  how='left', left_on=['year','state'], right_on = ['year','state'])
cPUA = pd.merge(cPU, rawAreaRepeat,  how='left', left_on=['year','state'], right_on = ['year','state'])

cPUA['pop_density_sq_mile'] = cPUA["population"]/cPUA['area_sq_mile']

cPUA.drop(['caveats',
          "state_abbr",
          "poverty_num",
          "unemployment_number",
          "employment_percent",
          "area_sq_mile",
          ], axis=1, inplace=True)


cPUA.drop_duplicates(keep='first', inplace=True)

print(cPUA)

cPUA.to_csv("cleaned_data", sep=',')