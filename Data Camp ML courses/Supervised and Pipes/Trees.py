####Decision trees
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test= train_test_split(X, y,
test_size=0.2,
stratify=y,
random_state=1)
###Random Tree
# =============================================================================
# dt = DecisionTreeClassifier(max_depth=8, random_state=1,criterion='gini')#criterion='entropy'
# dt.fit(X_train, y_train)
# y_pred = dt.predict(X_test)
# 
# acc = accuracy_score(y_test, y_pred)
# print("Test set accuracy: {:.2f}".format(acc))
# =============================================================================


###Regression Tree
# =============================================================================
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error as MSE
# dt = DecisionTreeRegressor(max_depth=18,
#              min_samples_leaf=0.01,
#             random_state=3)
# dt.fit(X_train, y_train)
# 
# y_pred = dt.predict(X_test)
# mse_dt = MSE(y_test, y_pred)
# rmse_dt = mse_dt**(1/2)
# print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
# 
# ####Check for Over/Under Fitting
# from sklearn.model_selection import  cross_val_score
# MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, 
#                                   scoring='neg_mean_squared_error', 
#                                   n_jobs=-1) 
# 
# RMSE_CV = (MSE_CV_scores.mean())**(1/2)
# 
# 
# print('CV RMSE: {:.2f}'.format(RMSE_CV))
# 
# y_pred_train = dt.predict(X_train)
# RMSE_train = (MSE(y_train, y_pred_train))**(1/2)
# print('Train RMSE: {:.2f}'.format(RMSE_train))
# =============================================================================

###RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt
import pandas as pd
rf = RandomForestRegressor(n_estimators=25 ,
            random_state=2)  
rf.fit(X_train, y_train) 
y_pred = rf.predict(X_test)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE of Random Forrest: {:.2f}'.format(rmse_test))

# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_)#add index = columns for better representation

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()