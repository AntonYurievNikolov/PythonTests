#Central Limit Theory
from numpy.random import randint
means = [randint(1, 7, 30).mean() for i in range(100)]
plt.hist(means)
plt.show()

#Distributions#########
#Bernoli Distribution
from scipy.stats import bernoulli
data = bernoulli.rvs(p=0.5, size=1000)
plt.hist(data)
plt.show()
#Binominal
from scipy.stats import binom
data = binom.rvs(n=10, p=0.8, size=1000)
plt.hist(data)
plt.show()
prob1 = binom.cdf(k=8, n=10, p=0.8)
print(prob1)
prob2 = binom.pmf(k=10, n=10, p=0.8)
print(prob2)

#Normal
# Generate normal data
from scipy.stats import norm
data = norm.rvs(size=1000)
plt.hist(data)
plt.show()
# Compute and print true probability for greater than 2
true_prob = 1 - norm.cdf(2)
print(true_prob)
# Compute and print sample probability for greater than 2
sample_prob = sum(obs > 2 for obs in data) / len(data)
print(sample_prob)


#EDA##########
#Encoding
# One-hot encode Company for laptops2
laptops2 = pd.get_dummies(data=laptops, columns=['Company'])
print(laptops2.head())

laptops.info()
sns.countplot(laptops.Company)
plt.show()

#Relations
#pairplot
# Generate the pair plot for the weather dataset
sns.pairplot(weather)
plt.show()
r = weather['Humidity9am'].corr(weather['Humidity3pm'])
r2 = r*r
print(r2)

#Confidence Intervals
from scipy.stats import sem, t
data = [1, 2, 3, 4, 5]
confidence = 0.95
std_err = sem(data)
margin_error = std_err * z_score
lower = sample_mean - margin_error
print(lower)
upper = sample_mean + margin_error
print(upper)

# Repeat this process 10 times 
heads = binom.rvs(50, 0.5, size=10)
for val in heads:
    confidence_interval = proportion_confint(val, 50, .10)
    print(confidence_interval)
    
#Z test
# Assign the number of conversions and total trials
num_control = results[results['Group'] == 'control']['Converted'].sum()
total_control = len(results[results['Group'] == 'control'])

# Assign the number of conversions and total trials
num_treat = results[results['Group'] == 'treatment']['Converted'].sum()
total_treat = len(results[results['Group'] == 'treatment'])

from statsmodels.stats.proportion import proportions_ztest
count = np.array([num_treat, num_control]) 
nobs = np.array([total_treat, total_control])

# Run the z-test and print the result 
stat, pval = proportions_ztest(count, nobs, alternative="larger")
print('{0:0.3f}'.format(pval))

#2 Tail T test
# Assign the prices of each group
asus = laptops[laptops['Company'] == 'Asus']['Price']
toshiba = laptops[laptops['Company'] == 'Toshiba']['Price']

# Run the t-test
from scipy.stats import ttest_ind
tstat, pval = ttest_ind(asus, toshiba)
print('{0:0.3f}'.format(pval))

#CALCULATING SAMPLE SIZE
# Standardize the effect size
from statsmodels.stats.proportion import proportion_effectsize
std_effect = proportion_effectsize(.20, .25)

# Assign and print the needed sample size
from statsmodels.stats.power import  zt_ind_solve_power
sample_size = zt_ind_solve_power(effect_size=std_effect, nobs1=None, alpha=.05, power=0.8)
print(sample_size)

sample_sizes = np.array(range(5, 100))
effect_sizes = np.array([0.2, 0.5, 0.8])

# Create results object for t-test analysis
from statsmodels.stats.power import TTestIndPower
results = TTestIndPower()
results.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
plt.show()

#P corrections
from statsmodels.sandbox.stats.multicomp import multipletests
pvals = [.01, .05, .10, .50, .99]
p_adjusted = multipletests(pvals, alpha=.05, method='bonferroni')
print(p_adjusted[0])
print(p_adjusted[1])

#Bias Variance Tradeoff visualization
# Generate and output the confusion matrix
from sklearn.metrics import confusion_matrix
preds = clf.predict(X_test)
matrix = confusion_matrix(y_test, preds)
print(matrix)
# Compute and print the precision
from sklearn.metrics import precision_score
preds = clf.predict(X_test)
precision = precision_score(y_test, preds)
print(precision)

#spot all null values
# Identify and print the the rows with null values
nulls = laptops[laptops.isnull().any(axis=1)]
print(nulls)