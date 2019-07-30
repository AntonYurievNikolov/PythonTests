###Classifiers
###EDA functions
#import seaborn as sns
#sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
###KNN
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.model_selection  import train_test_split
import numpy as np

digits = datasets.load_digits()
#print(digits.DESCR)
#print(digits.keys())
#print(digits.images.shape)
#print(digits.data.shape)
#plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()

X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=digits.target)

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
    
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#Metrics of the Classifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


###Regressions
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('gapminder.csv')

y = df['life'].values
X = df['fertility'].values

y = y.reshape(-1, 1)
X = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

reg = LinearRegression()
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

reg.fit(X_train, y_train)
y_pred = reg.predict(prediction_space)
print(reg.score(X, y))
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

###K-fold cross validation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
cv_scores = cross_val_score(reg,X,y,cv=5)

print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

###Reguralization
#Lasso
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.4,normalize=True)
lasso.fit(X , y)
lasso_coef =lasso.coef_ 
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(datasets.columns)), lasso_coef)
plt.xticks(range(len(datasets.columns)), datasets.columns.values, rotation=60)
plt.margins(0.02)
plt.show()

#Regularization II: Ridge
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []
ridge = Ridge(normalize=True)

for alpha in alpha_space:
    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))

display_plot(ridge_scores, ridge_scores_std)
