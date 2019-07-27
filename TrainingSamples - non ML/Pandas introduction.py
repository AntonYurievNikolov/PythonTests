print(df['2015'].count())

# Print the 5th and 95th percentiles
print(df.quantile([0.05,0.95]))

# Generate a box plot
years = ['1800','1850','1900','1950','2000']
df[years].plot(kind='box')
plt.show()

#Time Series
# Prepare a format string: time_format
time_format = '%Y-%m-%d %H:%M'
#.read_csv()
my_datetimes = pd.to_datetime(date_list, format=time_format)  
time_series = pd.Series(temperature_list, index=my_datetimes)
#if we want to add 2 time series
ts3 = ts2.reindex(ts1.index)

ts4 = ts2.reindex(ts1.index, method='ffill')

ts3 = ts2.reindex(ts1.index).interpolate(how='linear')
#downsampling
df2 = df['Temperature'].resample('D').count()
# Extract temperature data for August: august
august = df['Temperature']['2010-August']

august_highs = august.resample('D').max()
#smoothing
df['Temperature']['2010-Aug-01':'2010-Aug-15'].rolling(window=24).mean()
#timezones
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')

#plotting
df.Temperature['2010-06-10':'2010-06-17'].plot()
plt.show()
plt.clf()