import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import preprocessing

from xgboost import XGBRegressor
data_dir = Path('D:\\PythonTests\\Testing Kaggle\\19\\covid19-global-forecasting-week-3\\')
#
#xtrain = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv', parse_dates=['Date'])
#xtest = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv', parse_dates=['Date'])
#xsubmission = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')

train = pd.read_csv(data_dir/'train.csv', parse_dates=['Date'])
test = pd.read_csv(data_dir/'test.csv', parse_dates=['Date'])
submission = pd.read_csv(data_dir/'submission.csv')


train.rename(columns={'Country_Region':'Country'}, inplace=True)
test.rename(columns={'Country_Region':'Country'}, inplace=True)

train.rename(columns={'Province_State':'State'}, inplace=True)
test.rename(columns={'Province_State':'State'}, inplace=True)

train = train[train['Country'] == "Bulgaria"]
test= test[test['Country'] == "Bulgaria"]


y1_Train = train.iloc[:, -2]
y2_Train = train.iloc[:, -1]


NAN = "NAN"

def fillState(state, country):
    if state == NAN: return country
    return state


#Preprocessing
X_Train = train.copy()

X_Train['State'].fillna(NAN, inplace=True)
X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")
X_Train["Date"]  = X_Train["Date"].astype(int)



X_Test = test.copy()

X_Test['State'].fillna(NAN, inplace=True)
X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%m%d")
X_Test["Date"]  = X_Test["Date"].astype(int)


le = preprocessing.LabelEncoder()

X_Train.Country = le.fit_transform(X_Train.Country)
X_Train['State'] = le.fit_transform(X_Train['State'])

X_Test.Country = le.fit_transform(X_Test.Country)
X_Test['State'] = le.fit_transform(X_Test['State'])


countries = X_Train.Country.unique()


#The Model
out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
for country in countries:
    states = X_Train.loc[X_Train.Country == country, :].State.unique()
    for state in states:
        X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), ['State', 'Country', 'Date', 'ConfirmedCases', 'Fatalities']]
        
        y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']
        y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']
        
        X_Train_CS = X_Train_CS.loc[:, ['State', 'Country', 'Date']]
        
        X_Train_CS.Country = le.fit_transform(X_Train_CS.Country)
        X_Train_CS['State'] = le.fit_transform(X_Train_CS['State'])
        
        X_Test_CS = X_Test.loc[(X_Test.Country == country) & (X_Test.State == state), ['State', 'Country', 'Date', 'ForecastId']]
        
        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']
        X_Test_CS = X_Test_CS.loc[:, ['State', 'Country', 'Date']]
        
        X_Test_CS.Country = le.fit_transform(X_Test_CS.Country)
        X_Test_CS['State'] = le.fit_transform(X_Test_CS['State'])
        

        #After we transform them they should roughly follow linear regression trend
        y1_Train_CS = y1_Train_CS.apply(lambda x: np.log1p(x))
        y2_Train_CS = y2_Train_CS.apply(lambda x: np.log1p(x))
        model = linear_model.LinearRegression()

# Intentional overfit for Testing purposes    
#        model= XGBRegressor(
#                             reg_alpha=0, 
#                             reg_lambda=0,
#                             eta = 0.1, 
#                             n_estimators=5000,
#                             random_state =7)   
#        model= XGBRegressor(colsample_bytree=0.5,reg_alpha=1, reg_lambda=1, n_estimators=1000,random_state =7)
        
        xmodel1 = model
        xmodel1.fit(X_Train_CS, y1_Train_CS)
        y1_xpred = xmodel1.predict(X_Test_CS)
        
###        Need to add proper testing
#        test1 = xmodel1.predict(X_xTrain_CS)
#        print(np.sqrt(mean_squared_error(print, y1_xTrain_CS)))
#        0.08515638189555029
        xmodel2 = model
        xmodel2.fit(X_Train_CS, y2_Train_CS)
        y2_xpred = xmodel2.predict(X_Test_CS)
        
#        test2 =   xmodel2.predict(X_xTrain_CS)
#        print(np.sqrt(mean_squared_error(test2, y2_xTrain_CS)))
        
        xdata = pd.DataFrame({'ForecastId': X_Test_CS_Id, 
                              'ConfirmedCases': np.expm1(y1_xpred) , 
                              'Fatalities': np.expm1(y2_xpred)})
        out = pd.concat([out, xdata], axis=0)
        
out.ForecastId = out.ForecastId.astype('int')
out.tail()
out.to_csv('submission.csv', index=False)