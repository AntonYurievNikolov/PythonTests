import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import preprocessing

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from xgboost import XGBRegressor
data_dir = Path('D:\\PythonTests\\Testing Kaggle\\19\\covid19-global-forecasting-week-3\\')
#data_dir = '/kaggle/input/covid19-global-forecasting-week-3/'
# Load Data
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
        n_cols =X_Train_CS.shape[1]
#        Tranform the data
#        y1_Train_CS = y2_Train_CS.apply(lambda x: np.log1p(x))
#        y2_Train_CS = y2_Train_CS.apply(lambda x: np.log1p(x))
        model = Sequential()
        model.add(layers.Dense(4, input_shape = (n_cols,)))
        model.add(layers.Dense(2, activation='relu' ))
        model.add(layers.Dense(1))
        
        model.compile(optimizer=Adam(lr=0.001),loss='mean_squared_error',metrics=['mse'])
        
        model2 = model
        model.fit(X_Train_CS, y1_Train_CS,epochs=100)
#     
        y1_pred = model.predict(X_Test_CS)
        
        model2.fit(X_Train_CS, y2_Train_CS,epochs=100)

        y2_pred = model2.predict(X_Test_CS)
        
        xdata = pd.DataFrame({'ForecastId': X_Test_CS_Id, 
                              'ConfirmedCases': y1_pred[:,0] , 
                              'Fatalities': y2_pred[:,0]})
        out = pd.concat([out, xdata], axis=0)
        
out.ForecastId = out.ForecastId.astype('int')
out.tail()
out.to_csv('submission.csv', index=False)