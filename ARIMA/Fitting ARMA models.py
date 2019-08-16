#ARMA
model = ARMA(earthquake, order=(3,1))
results = model.fit()
print(results.summary())
#ARMAX
# Instantiate the model
model = ARMA(hospital['wait_times_hrs'], order=(2,1), 
             exog=hospital['nurse_count'])
results = model.fit()
# Print model fit summary
print(results.summary())

# Generate predictions
one_step_forecast = results.get_prediction(start=-30)
# Extract prediction mean
mean_forecast = one_step_forecast.predicted_mean
# Get confidence intervals of predictions
confidence_intervals = one_step_forecast.conf_int()
# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower close']
upper_limits = confidence_intervals.loc[:,'upper close']
# Print best estimate predictions
print(mean_forecast)

# plot the amazon data
plt.plot(amazon.index, amazon, label='observed')

# plot your mean  predictions
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits, 
               upper_limits, color='pink')

# set labels, legends and show plot
plt.xlabel('Date')
plt.ylabel('Amazon Stock Price - Close USD')
plt.legend()
plt.show()

#SAME BUT DYNAMIC, NOT 1 STEP
# Generate Dynamic predictions
dynamic_forecast = results.get_prediction(start=-30, dynamic=True)

# Extract prediction mean
mean_forecast = dynamic_forecast.predicted_mean

# Get confidence intervals of predictions
confidence_intervals = dynamic_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower close']
upper_limits = confidence_intervals.loc[:,'upper close']

# Print best estimate predictions
print(mean_forecast)

#SAME BUT WITH SARIMAX
# Create ARIMA(2,1,2) model
arima = SARIMAX(amazon, order=(2,1,2))
arima_results = arima.fit()
arima_value_forecast = arima_results.get_forecast(steps=10).predicted_mean
print(arima_value_forecast)