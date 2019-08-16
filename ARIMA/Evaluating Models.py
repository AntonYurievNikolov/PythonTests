#AutoCorrelationFunction ACF and PartialACF
# Import
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
plot_acf(df, lags=10, zero=False, ax=ax1)
plot_pacf(df, lags=10, zero=False, ax=ax2)
plt.show()
#        AR(p)	                  MA(q)	                       ARMA(p,q)
#ACF	 Tails off	              Cuts off after lag q	       Tails off
#PACF	 Cuts off after lag p	  Tails off	                   Tails off

#AIC(AKAIKE INFORMATION CRITERIA) and BIC(Baysian IC
order_aic_bic=[]
for p in range(3):
    for q in range(3):
        model = SARIMAX(df, order=(p,0,q))
        results = model.fit()
        order_aic_bic.append((p,q,results.aic, results.bic)))
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p', 'q', 'AIC', 'BIC'])
print(order_df.sort_values('AIC'))
print(order_df.sort_values('BIC'))

#COMMON MODEL DIAGNOSTIC
mae = np.mean(np.abs(results.resid))
print(mae)

#
#Test	Null hypothesis	P-value name
#Ljung-Box	There are no correlations in the residual
#Prob(Q)
#Jarque-Bera	The residuals are normally distributed	Prob(JB)
# Create and fit model
model1 = SARIMAX(df, order=(3,0,1))
results1 = model1.fit()

# Print summary
print(results1.summary())


#
#Test	                        Good fit
#Standardized residual	      There are no obvious patterns in the residuals
#Histogram plus kde estimate	    The KDE curve should be very similar to the normal distribution
#Normal Q-Q	                     Most of the data points should lie on the straight line
#Correlogram	                    95% of correlations for lag greater than one should not be significant

# Create and fit model
model = SARIMAX(df, order=(1,1,1))
results = model.fit()
results.plot_diagnostics()
plt.show()

#BOX JENKINKS METHOD
# Plot time series
savings.plot()
plt.show()

# Run Dicky-Fuller test
result = adfuller(savings['savings'])
# Print test statistic
print(result[0])
# Print p-value
print(result[1])

# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of savings on ax1
plot_acf(savings, lags=10, zero=False, ax=ax1)

# Plot the PACF of savings on ax2
plot_pacf(savings, lags=10, zero=False, ax=ax2)

plt.show()

# Loop over p values from 0-3
for p in range(4):
  
  # Loop over q values from 0-3
    for q in range(4):
      try:
        # Create and fit ARMA(p,q) model
        model = SARIMAX(savings, order=(p,0,q), trend='c')
        results = model.fit()
        
        # Print p, q, AIC, BIC
        print(p, q, results.aic, results.bic)
        
      except:
        print(p, q, None, None)

# Create and fit model
model = SARIMAX(savings, order=(1,0,2), trend='c')
results = model.fit()

# Create the 4 diagostics plots
results.plot_diagnostics()
plt.show()

# Print summary
print(results.summary())