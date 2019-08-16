import matplotlib.pyplot as plt
import pandas as pd
candy = pd.read_csv('candy_production.csv', 
            index_col='date',
            parse_dates=True)
fig, ax = plt.subplots()
candy.plot(ax=ax)
plt.show()
candy_train = candy.loc[:'2006']
candy_test = candy.loc['2007':]
fig, ax = plt.subplots()
candy_train.plot(ax=ax)
candy_test.plot(ax=ax)
plt.show()

#TEST stationary
# Calculate the second difference of the time series
city_stationary = city.diff().dropna().diff().dropna()
# Run ADF test on the differenced time series
result = adfuller(city_stationary['city_population'])
# Plot the differenced time series
fig, ax = plt.subplots()
city_stationary.plot(ax=ax)
plt.show()
# Print the test statistic and the p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])

amazon_diff = amazon.diff().dropna()
result_diff = adfuller(amazon_diff['close'])
print(result_diff)
amazon_log = np.log((amazon/amazon.shift(1)).dropna())
result_log = adfuller(amazon_log['close'])
print(result_log)

#Generating ARMA data
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(3)

# Set coefficients
ar_coefs = [1, 0.2]
ma_coefs = [1, 0.3, 0.4]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, sigma=0.5, )

plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()