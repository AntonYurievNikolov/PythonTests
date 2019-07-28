# Count the number of missing values in each column
print(ri.isnull()..sum())

ri.drop([county_name, state], axis='columns', inplace=True)
#droping rows
ri.dropna(subset=['driver_gender'], inplace=True)
#cast
ri['is_arrested'] = ri.is_arrested.astype('bool')
#ughhh pandas needs to manually clean the date time...
combined = ri.stop_date.str.cat(ri.stop_time, sep=' ')

# Convert 'combined' to datetime format
ri['stop_datetime'] = pd.to_datetime(combined)
ri.set_index('stop_datetime', inplace=True)
#see as % 
print(ri.violation.value_counts(normalize=True))

#gr
print(ri.groupby('driver_gender').search_conducted.mean())
#contains str
ri.search_type.str.contains('Protective Frisk', na=False)
#getting a part of time index
print(ri.groupby(ri.index.hour).is_arrested.mean())
#resampling
ri.drugs_related_stop.resample('A').mean()

#Crosstab for Frequency table !!!
pd.crosstab(ri.district, ri.violation)

#mapping
# Create a dictionary that maps strings to integers
mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}

# Convert the 'stop_duration' strings to integers using the 'mapping'
ri['stop_minutes'] = ri.stop_duration.map(mapping)

# Create a list of weather ratings in logical order
cats = ['good', 'bad', 'worse']

# Change the data type of 'rating' to category
weather['rating'] = weather.rating.astype('category', ordered=True, categories=cats)