from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = datasets.load_iris()
X = pd.DataFrame(data.data)
y = data.target

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y


x_vers, y_vers = ecdf(X.iloc[: , 1])
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.show()
#PMF
#CDF
#Comparison
