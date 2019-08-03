# Import xgboost
import xgboost as xgb
import matplotlib as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pandas as pd
import numpy as np

data = datasets.load_boston()
X = data.data
y = data.target

housing_dmatrix = xgb.DMatrix(data=X, label=y)

params = {"objective":"reg:squarederror", "max_depth":3}
num_rounds = [5, 10, 15, 100]
final_rmse_per_round = []
for curr_num_rounds in num_rounds:
    cv_results = xgb.cv(dtrain=housing_dmatrix, 
                        params=params, 
                        nfold=3, 
                        num_boost_round=curr_num_rounds, 
                        metrics="rmse", 
                        as_pandas=True, 
                        seed=123)
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))


####Early Stopping
cv_results = xgb.cv(dtrain=housing_dmatrix, 
                        params=params, 
                        nfold=3, 
                        early_stopping_rounds=50, 
                        metrics="rmse", 
                        as_pandas=True, 
                        seed=123)
print(cv_results)

####Tunnable parameters
#learning rate: learning rate/eta
#gamma: min loss reduction to create new tree split
#lambda: L2 reg on leaf weights
#alpha: L1 reg on leaf weights
#max_depth: max depth per tree
#subsample: % samples used per tree
#colsample_bytree: % features used per tree

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]

}
params = {"objective":"reg:squarederror"}
gbm = xgb.XGBRegressor(objective="reg:squarederror")    # to remove deprecated warnings)
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid,
                        scoring='neg_mean_squared_error', cv=4, verbose=1)
grid_mse.fit(X, y)
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

#Random Search
# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}

gbm = xgb.XGBRegressor(n_estimators=10,objective="reg:squarederror")
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid,
                                    n_iter=5, scoring='neg_mean_squared_error', cv=4, verbose=1)
randomized_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ",randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))