# Import xgboost
import xgboost as xgb
import matplotlib as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pandas as pd
import numpy as np

data = datasets.load_breast_cancer()
X = data.data
y = data.target


#Classification
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
xg_cl.fit (X_train, y_train)
preds = xg_cl.predict(X_test)

#accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

dmatrix = xgb.DMatrix(data=X_train, label=y_train)
#test-error-mean
params = {"objective":"reg:logistic", "max_depth":3}
cv_results = xgb.cv(dtrain=dmatrix, 
                    params=params, nfold=3, 
                    num_boost_round=5, 
                    metrics="error", 
                    as_pandas=True, 
                    seed=123)
print(cv_results)
print(((1-cv_results["test-error-mean"]).iloc[-1]))

#AUC test
cv_results = xgb.cv(dtrain=dmatrix, 
                    params=params, 
                    nfold=3, 
                    num_boost_round=5, 
                    metrics="auc", 
                    as_pandas=True, 
                    seed=123)
print(cv_results)
print((cv_results["test-auc-mean"]).iloc[-1])

###Regression
data = datasets.load_boston()
X = data.data
y = data.target
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

DM_train= xgb.DMatrix(data=X_train, label=y_train)
DM_test= xgb.DMatrix(data=X_test, label=y_test)


xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10, seed=123)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test =  xgb.DMatrix(data=X_test, label=y_test)
params = {"booster":"gblinear", "objective":"reg:squarederror"}
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)
preds = xg_reg.predict(DM_test)
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))


housing_dmatrix = xgb.DMatrix(data=X,label=y)
params = {"objective":"reg:squarederror", "max_depth":4}
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
print(cv_results)
print((cv_results["test-rmse-mean"]).tail(1))


####REGULARIZATION
housing_dmatrix = xgb.DMatrix(data=X, label=y)
reg_params = [1, 0.5, 0.005]
params = {"objective":"reg:squarederror","max_depth":3}
rmses_l2 = []
for reg in reg_params:
    #L1
    params["alpha"] = reg
    #L2
    params["lambda"] = reg
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2","rmse"]))


###VIZUALIZATION
#xgb.plot_tree(xg_reg, num_trees=0)
#plt.show()
#xgb.plot_tree(xg_reg, num_trees=4)
#plt.show()
#xgb.plot_tree(xg_reg, num_trees=9, rankdir='LR')
#plt.show()
#xgb.plot_importance(xg_reg)
