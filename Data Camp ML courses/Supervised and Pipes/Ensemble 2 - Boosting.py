###DATA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors  import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier#,AdaBoostRegressor
from sklearn.metrics import mean_squared_error as MSE

dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test= train_test_split(X, y,
test_size=0.2,
stratify=y,
random_state=1)

SEED=1

####ADABOOST for Classification #,AdaBoostRegressor for regression
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

ada.fit(X_train, y_train)
y_pred_proba = ada.predict_proba(X_test)[:,1]
y_pred = ada.predict(X_test)

from sklearn.metrics import roc_auc_score
ada_roc_auc = roc_auc_score(y_test, y_pred_proba)
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))
print('Adaboost Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))

###Gradient Boosting for Regression
# =============================================================================
# from sklearn.ensemble import GradientBoostingRegressor
# gb = GradientBoostingRegressor(max_depth=4,
#                                n_estimators=200,
#                                random_state=2)
# gb.fit(X_train, y_train)
# y_pred = gb.predict(X_test)
# 
# mse_test = MSE(y_test, y_pred)
# rmse_test = mse_test**(1/2)
# print('Test set RMSE of gb: {:.3f}'.format(rmse_test))
# =============================================================================


###Stogastic GBR
from sklearn.ensemble import GradientBoostingRegressor
sgbr = GradientBoostingRegressor(max_depth=4, 
                                 subsample=0.9,
                                 max_features=0.75,
                                 n_estimators=200,                                
                                 random_state=2)

sgbr.fit(X_train, y_train)
y_pred = sgbr.predict(X_test)
mse_test = MSE(y_test, y_pred)
rmse_test = mse_test**(1/2)
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))