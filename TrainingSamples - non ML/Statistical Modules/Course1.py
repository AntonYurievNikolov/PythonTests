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
_=plt.plot(x_vers,y_vers,marker = '.',linestyle = 'none')
# Label the axes
_=plt.xlabel("petal length (cm)")
_=plt.ylabel('ECDF')
_=plt.show()

#Percentiles and Boxplots wiht 1.5IQR = Outlier
np.percentile(versicolor_petal_length, np.array([2.5, 25, 50, 75, 97.5]
#ECFG vs Percentiles
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
       linestyle='none')
plt.show()
#Box Plots
_ = sns.boxplot(x='species',y='petal length (cm)',data=df)
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')
plt.show()

#Variance + Standard Deviation
#np.var/std...
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)
#Pierson's Correlation
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    corr_mat = np.corrcoef(x, y)
    return corr_mat[0,1]

r = pearson_r(versicolor_petal_width, versicolor_petal_length)

#!!!"hacker" statistics!!!
np.random.seed(42)
random_numbers = np.empty(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()
_ = plt.hist(random_numbers)s
plt.show()

def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    n_success = 0
    for i in range(n):
        random_number = np.random.random()
        if random_number < p:
            n_success += 1
    return n_success

n_defaults=np.empty(1000)
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100,0.05)
    
#BInomial Distr
n_defaults=np.random.binomial(n = 100 , p = 0.05, size=10000)
x, y = ecdf(n_defaults)
_=plt.plot(x,y,marker = '.',linestyle = 'none')
_=plt.xlabel("number of defaults out of 100 loans")
_=plt.ylabel('ECDF')
_=plt.show()

#Poisson
samples_poisson = np.random.poisson(10, size=10000)

#Normal distribution and checking whether your data is!!!
# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)
# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size=10000)
# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers)
# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()

#Exponential Distribution
def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size=size)
    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size=size)
    return t1 + t2

waiting_times=successive_poisson(764,715,100000)
# Make the histogram PDF
_=plt.hist(waiting_times,bins=100, normed=True,  histtype='step')
_=plt.xlabel('total waiting time (games)')
_=plt.ylabel('PDF')
_=plt.show()

