from pathlib import Path
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def RMSLE(pred,actual):
    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))

data_dir = Path('D:\\PythonTests\\Testing Kaggle\\19\\covid19-global-forecasting-week-4\\')

pd.set_option('mode.chained_assignment', None)
train = pd.read_csv(data_dir/'train.csv')
test = pd.read_csv(data_dir/'test.csv')
train['Province_State'].fillna('', inplace=True)
test['Province_State'].fillna('', inplace=True)
train['Date'] =  pd.to_datetime(train['Date'])
test['Date'] =  pd.to_datetime(test['Date'])
train = train.sort_values(['Country_Region','Province_State','Date'])
test = test.sort_values(['Country_Region','Province_State','Date'])


train = train[train['Country_Region'] == "Bulgaria"]
test= test[test['Country_Region'] == "Bulgaria"]


feature_day = [1,20,50,100,200,500,1000]
def CreateInput(data):
    feature = []
    for day in feature_day:
        #Get information in train data
        data.loc[:,'Number day from ' + str(day) + ' case'] = 0
        if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):
            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        
        else:
            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].min()       
        for i in range(0, len(data)):
            if (data['Date'].iloc[i] > fromday):
                day_denta = data['Date'].iloc[i] - fromday
                data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 
        feature = feature + ['Number day from ' + str(day) + ' case']
    
    return data[feature]

pred_data_all = pd.DataFrame()
for country in train['Country_Region'].unique():
    for province in train[(train['Country_Region'] == country)]['Province_State'].unique():
        df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]
        df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
        X_train = CreateInput(df_train)
        y_train_confirmed = df_train['ConfirmedCases'].ravel()
        y_train_fatalities = df_train['Fatalities'].ravel()
        X_pred = CreateInput(df_test)
        
        # Only train above 50 cases
        for day in sorted(feature_day,reverse = True):
            feature_use = 'Number day from ' + str(day) + ' case'
            idx = X_train[X_train[feature_use] == 0].shape[0]     
            if (X_train[X_train[feature_use] > 0].shape[0] >= 20):
                break
                                           
        adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)
        adjusted_y_train_confirmed = y_train_confirmed[idx:]
        adjusted_y_train_fatalities = y_train_fatalities[idx:] #.values.reshape(-1, 1)
        idx = X_pred[X_pred[feature_use] == 0].shape[0]    
        adjusted_X_pred = X_pred[idx:][feature_use].values.reshape(-1, 1)
        
        pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]
        max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].max()
        min_test_date = pred_data['Date'].min()
        model = SARIMAX(adjusted_y_train_confirmed, order=(1,1,0), 
                        #seasonal_order=(1,1,0,12),
                        measurement_error=True).fit(disp=False)
        y_hat_confirmed = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])
        y_train_confirmed = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['ConfirmedCases'].values
        y_hat_confirmed = np.concatenate((y_train_confirmed,y_hat_confirmed), axis = 0)
               
        model = SARIMAX(adjusted_y_train_fatalities, order=(1,1,0), 
                        #seasonal_order=(1,1,0,12),
                        measurement_error=True).fit(disp=False)
        y_hat_fatalities = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])
        y_train_fatalities = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['Fatalities'].values
        y_hat_fatalities = np.concatenate((y_train_fatalities,y_hat_fatalities), axis = 0)
        
        
        pred_data['ConfirmedCases_hat'] =  y_hat_confirmed
        pred_data['Fatalities_hat'] = y_hat_fatalities
        pred_data_all = pred_data_all.append(pred_data)

df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')
df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0
df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0
df_val_2 = df_val.copy()


#method_list = ['Exponential Smoothing','SARIMA']
#method_val = [df_val_1,df_val_2]
#for i in range(0,2):
#    df_val = method_val[i]
#    method_score = [method_list[i]] + [RMSLE(df_val[(df_val['ConfirmedCases'].isnull() == False)]['ConfirmedCases'].values,df_val[(df_val['ConfirmedCases'].isnull() == False)]['ConfirmedCases_hat'].values)] + [RMSLE(df_val[(df_val['Fatalities'].isnull() == False)]['Fatalities'].values,df_val[(df_val['Fatalities'].isnull() == False)]['Fatalities_hat'].values)]
#    print (method_score)


df_val = df_val_2
submission = df_val[['ForecastId','ConfirmedCases_hat','Fatalities_hat']]
submission.columns = ['ForecastId','ConfirmedCases','Fatalities']
submission.to_csv('submission.csv', index=False)
submission
