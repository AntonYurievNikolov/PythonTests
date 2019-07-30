import scipy.stats
import statsmodels

#cont. Exponential
np.random.seed(42)
tau = nohitter_times.mean()
inter_nohitter_time = np.random.exponential(tau, 100000)
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')
plt.show()

#Check whether it follows normal
x, y = ecdf(nohitter_times)
x_theor, y_theor = ecdf(inter_nohitter_time)
# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')
plt.show()

#Linear regression
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')
a, b = np.polyfit(illiteracy,fertility,deg=1)
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')
x = np.array([0,100])
y = a * x + b
_ = plt.plot(x, y)
plt.show()

#SAMPLING WITH REPLACMENTS AND BOOTSTRAPING, Confidence intervals
for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()

#Functions
def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data,func)
    return bs_replicates


bs_replicates = draw_bs_reps(rainfall,np.mean,10000)
# Compute and print SEM!!!!!!!
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)
#computing the bootstrap
bs_std = np.std(bs_replicates)
print(bs_std)

_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')
plt.show()

#Conf Interval
bs_replicates = draw_bs_reps(nohitter_times,np.mean,10000)
# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates,[2.5,97.5])
print('95% confidence interval =', conf_int, 'games')

#PAIRED BOOTSTRAPPING
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""
    inds = np.arange(len(x))
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps

bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy,fertility,1000)
print(np.percentile(bs_slope_reps, [2.5,97.5]))

x = np.array([0, 100])
for i in range(100):
    _ = plt.plot(x, 
                 bs_slope_reps[i] * x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()