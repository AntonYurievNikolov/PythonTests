# Import xgboost
import xgboost as xgb
import matplotlib as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,cross_val_score, train_test_split 
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
import pandas as pd
import numpy as np




df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")



categorical_mask = (df.dtypes == object)
categorical_columns = df.columns[categorical_mask].tolist()
non_categorical_columns = df.columns[~categorical_mask].tolist()


#X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)
#Quick and durty - this will be extended with meaningfull fill for the missing data
df[categorical_columns] = df[categorical_columns].fillna('')
df[non_categorical_columns] = df[non_categorical_columns].fillna(0)


X = df.iloc[:,:-1]
y =  df.iloc[:,-1]
#nulls_per_column = X.isnull().sum()
#print(nulls_per_column)
#print(nulls_per_column.sum())

numeric_imputation_mapper = DataFrameMapper(
                                            [([numeric_feature],SimpleImputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df=True,
                                            df_out=True
                                           )
categorical_imputation_mapper = DataFrameMapper(
                                                [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                input_df=True,
                                                df_out=True
                                               )

numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)
                                         ])

transformers = []
transformers.extend([([numeric_feature], [SimpleImputer(strategy="median"), 
                                  StandardScaler()]) for numeric_feature in non_categorical_columns])
transformers.extend([(category_feature , [CategoricalImputer()]) for category_feature in categorical_columns])

num_cat_union = DataFrameMapper(transformers,
                                input_df=True,
                                df_out=True)
# Create full pipeline
pipeline = Pipeline([
#                     ("featureunion", num_cat_union),
#                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
#                     ("std_scaler", StandardScaler(with_mean=False)),
                     ("clf", xgb.XGBRegressor(objective = "reg:squarederror"))
                    ])

cross_val_scores = cross_val_score(pipeline, 
                                    X.to_dict("records"), 
                                    y, 
                                    cv=3, 
                                    scoring="neg_mean_squared_error")
print("3-fold AUC: ", np.mean(cross_val_scores))


# =============================================================================
# # Create the parameter grid
# gbm_param_grid = {
#     'clf__learning_rate': np.arange(.05, 1, .05),
#     'clf__max_depth': np.arange(3,10, 1),
#     'clf__n_estimators': np.arange(50, 200, 50)
# }
# 
# # Perform RandomizedSearchCV
# randomized_neg_mean_squared_error = RandomizedSearchCV(estimator=pipeline,
#                                         param_distributions=gbm_param_grid,
#                                         n_iter=2, scoring='neg_mean_squared_error', cv=2, verbose=1)
# 
# # Fit the estimator
# randomized_neg_mean_squared_error.fit(X, y)
# 
# # Compute metrics
# print(randomized_neg_mean_squared_error.best_score_)
# print(randomized_neg_mean_squared_error.best_estimator_)
# =============================================================================

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

  #Working Test with Pipes
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
           ("xgb_model", xgb.XGBRegressor(objective = "reg:squarederror"))]
  
xgb_pipeline = Pipeline(steps)
  #xgb_pipeline.fit(X.to_dict("records"), y)
  #predict = xgb_pipeline.predict(X.to_dict("records"))
  
  
cross_val_scores = cross_val_score(xgb_pipeline, 
                                     X.to_dict("records"), 
                                     y, 
                                     cv=2, 
                                     scoring="neg_mean_squared_error")
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))
