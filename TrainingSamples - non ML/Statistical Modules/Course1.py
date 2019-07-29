#preventing Binning Bias with SwarmPlots
_ = sns.swarmplot(x='species', y='petal length (cm)', data=df)
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')
plt.show()

#Empirical cumulative distribution functions=ECDF
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y
x_vers, y_vers = ecdf(versicolor_petal_length)
plt.plot(x_vers,y_vers,marker = '.',linestyle = 'none')
# Label the axes
plt.xlabel("petal length (cm)")
plt.ylabel('ECDF')
plt.show()