from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(milk_production['pounds_per_cow'], 
                            freq=12)
decomp.plot()
plt.show()
# Create figure and subplot
fig, ax1 = plt.subplots()
plot_acf(water['water_consumers'], lags=25, zero=False,  ax=ax1)
plt.show()


water_2 = water - water.rolling(15).mean()
water_2 =  water_2.dropna()
fig, ax1 = plt.subplots()
plot_acf(water_2['water_consumers'], lags=25, zero=False, ax=ax1)
plt.show()

#SEASONAL ARMINA - SARIMA
# Create a SARIMAX model
model = SARIMAX(df1, order=(1,0,0), seasonal_order=(1,1,0,7))
results = model.fit()
print(results.summary())

#Finding the order
# Take the first and seasonal differences and drop NaNs
aus_employment_diff = aus_employment.diff().diff(12).dropna()

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))
plot_acf(aus_employment_diff, ax=ax1,lags = 11, zero = False)
plot_pacf(aus_employment_diff, ax = ax2, lags = 11, zero = False)

plt.show()
#AR(p)	MA(q)	ARMA(p,q)
#ACF	Tails off	Cuts off after lag q	Tails off
#PACF	Cuts off after lag p	Tails off	Tails off
# Make list of lags
lags = [12, 24, 36, 48, 60]
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))
plot_acf(aus_employment_diff, lags=lags, ax=ax1)
plot_pacf(aus_employment_diff, lags=lags, ax=ax2)

plt.show()


#Compare the forecast
arima_pred = arima_results.get_forecast(steps=25)
arima_mean = arima_pred.predicted_mean
sarima_pred = sarima_results.get_forecast(steps=25)
sarima_mean = sarima_pred.predicted_mean
plt.plot(dates, sarima_mean, label='SARIMA')
plt.plot(dates, arima_mean, label='ARIMA')
plt.plot(wisconsin_test, label='observed')
plt.legend()
plt.show()

#SERCHING FOR OPTIMAL MODEL ORDER
import pmdarima as pm
# Create auto_arima model
# Create model for SARIMAX(p,0,q)(P,1,Q)7
model3 = pm.auto_arima(df3,
                      seasonal=True, m=7,
                      d=1, D=1, 
                      start_p=1, start_q=1,
                      max_p=1, max_q=1,
                      max_P=1, max_Q=1,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Print model summary
print(model3.summary())

# Import joblib
import joblib
filename = "candy_model.pkl"
joblib.dump(model,"candy_model.pkl")
loaded_model = joblib.load(filename)
loaded_model.update(df_new)

results.plot_diagnostics()
plt.show()

# Create forecast object
forecast_object = results.get_forecast(steps=136)
mean = forecast_object.predicted_mean
conf_int = forecast_object.conf_int()
dates = mean.index
plt.figure()

# Plot past CO2 levels
plt.plot(co2.index.values, co2.values, label='past')
plt.plot(dates, mean, label='predicted')
plt.fill_between(dates, conf_int.iloc[:,0], conf_int.iloc[:,1], alpha=0.2)
plt.legend()
plt.show()

# Print last predicted mean
print(mean.iloc[-1])

# Print last confidence interval
print(conf_int.iloc[-1])