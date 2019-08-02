# Import xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split 
from sklearn import datasets
import pandas as pd
import numpy as np

data = datasets.load_breast_cancer()
X = data.data
y = data.target


# Create the training and test sets
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