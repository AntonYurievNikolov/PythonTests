# Import xgboost
import xgboost as xgb
import matplotlib as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,cross_val_score, train_test_split 
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn_pandas import DataFrameMapper
# from sklearn_pandas import CategoricalImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
import pandas as pd
import numpy as np

import os
cwd = os.getcwd()

#
#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')

train = pd.read_csv("C:\\Work\\Python\\PythonTests\\XGBOOST\\house-prices-advanced-regression-techniques\\train.csv")
test = pd.read_csv("C:\\Work\\Python\\PythonTests\\XGBOOST\\house-prices-advanced-regression-techniques\\test.csv")
#We do not need the ID column
train_ID = train['Id']
test_ID = test['Id']
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


categorical_mask = (train.dtypes == object)
categorical_columns = train.columns[categorical_mask].tolist()
non_categorical_columns = train.columns[~categorical_mask].tolist()


#X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)
#Quick and durty - this will be extended with meaningfull fill for the missing data

train[categorical_columns] = train[categorical_columns].fillna('')
#train[non_categorical_columns] = train[non_categorical_columns].fillna(train.mean())

####Qick Test Remove later
#Remove ouliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
#Normilize the Target
#train["SalePrice"] = np.log1p(train["SalePrice"])

X = train.iloc[:,:-1]#pd.get_dummies() 
y =  train.iloc[:,-1]
#nulls_per_column = X.isnull().sum()
#print(nulls_per_column)
#print(nulls_per_column.sum())
###BUILD THE PIPE
get_text_data = FunctionTransformer(lambda x: x[categorical_columns], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[non_categorical_columns], validate=False)

xgbreg=xgb.XGBRegressor(     colsample_bytree=0.4603, gamma=0.06, 
                             learning_rate=0.05, max_depth=4, 
                             min_child_weight=1.8, n_estimators=2200,
                             reg_alpha=0.47, reg_lambda=0.86,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', SimpleImputer()),
                    ("std_scaler", StandardScaler(with_mean=False))
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
#                    ('OneHot', pd.get_dummies )
                ]))
             ]
        )

pipeline = Pipeline([
        ('union', process_and_join_features),
         ("ohe_onestep", DictVectorizer(sort=False)),
        ('clf', xgbreg)
    ])



## Create full pipeline
pipeline = Pipeline([
                     ("ohe_onestep", DictVectorizer(sort=False)),
                     ("clf",xgbreg)
                    ])
cv = 3
cross_val_scores = cross_val_score(pipeline, 
                                    X.to_dict("records"), 
                                    y, 
                                    cv=cv, 
                                    scoring="neg_mean_squared_error")
print(cv,"-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))


# Create the parameter grid
#gbm_param_grid = {
#     'clf__learning_rate': np.arange(.05, 1, .05),
#     'clf__max_depth': np.arange(3,10, 1),
#     'clf__n_estimators': np.arange(50, 200, 50)
# }
# 
## Perform RandomizedSearchCV
#randomized_neg_mean_squared_error = RandomizedSearchCV(estimator=pipeline,
#                                         param_distributions=gbm_param_grid,
#                                         n_iter=9, scoring='neg_mean_squared_error', cv=2, verbose=1)
# 
## Fit the estimator
#randomized_neg_mean_squared_error.fit(X.to_dict("records"), y)
# 
## Compute metrics
#print(np.mean(np.sqrt(np.abs(randomized_neg_mean_squared_error.best_score_))))
#print(randomized_neg_mean_squared_error.best_estimator_)

##Encode the Categorical Data
#print(df[categorical_columns].head())
#le = LabelEncoder()
#
#df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x.fillna('')))
#print(df[categorical_columns].head())
#
##  OneHotEncoder
#ohe = OneHotEncoder(categorical_features=categorical_mask, sparse=False)
#df_encoded = ohe.fit_transform(df[categorical_columns].fillna(''))
#print(df_encoded[:5, :])
#print(df.shape)
#print(df_encoded.shape)

#DictVectorizer
#df_dict = df.to_dict("records")
#dv = DictVectorizer(sparse=False)
#df_encoded = dv.fit_transform(df_dict)
#print(df_encoded[:5,:])
#print(dv.vocabulary_)

