
import numpy as np
import pandas as pd
import math


import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
data_dir = Path('D:\\PythonTests\\Testing Kaggle\\19\\covid19-global-forecasting-week-1')

data = pd.read_csv(data_dir/'train.csv', parse_dates=['Date'])

data.rename(columns={'Date': 'date', 
                     'Id': 'id',
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Lat':'lat',
                     'Long': 'long',
                     'ConfirmedCases': 'confirmed',
                     'Fatalities':'deaths',
                    }, inplace=True)

data.head()
cleaned_data = pd.read_csv('D:\\PythonTests\\Testing Kaggle\\19\\covid_19_clean_complete.csv', parse_dates=['Date'])
cleaned_data.head()
cleaned_data.rename(columns={'ObservationDate': 'date', 
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Last Update':'last_updated',
                     'Confirmed': 'confirmed',
                     'Deaths':'deaths',
                     'Recovered':'recovered'
                    }, inplace=True)

# cases 
cases = ['confirmed', 'deaths', 'recovered', 'active']
cleaned_data['active'] = cleaned_data['confirmed'] - cleaned_data['deaths'] - cleaned_data['recovered']
cleaned_data['country'] = cleaned_data['country'].replace('Mainland China', 'China')

# filling missing values 
cleaned_data[['state']] = cleaned_data[['state']].fillna('')
cleaned_data[cases] = cleaned_data[cases].fillna(0)
cleaned_data.rename(columns={'Date':'date'}, inplace=True)

data = cleaned_data

grouped = data.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()
#enchance the data
grouped["newCases"] = grouped["confirmed"] - grouped["confirmed"].shift()
grouped["newCases"] = grouped["newCases"].fillna(method='backfill')
grouped["newCasesRolling"] = grouped.rolling(7,1)["newCases"].mean()

#check the Trend 

ax = sns.lineplot(y="newCasesRolling", x="confirmed", data=grouped)
#Bulgaria 
grouped_bulgaria = data[data['country'] == "Bulgaria"].reset_index()
grouped_bulgaria_date = grouped_bulgaria.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

#enchance the data
grouped_bulgaria_date["newCases"] = grouped_bulgaria_date["confirmed"] - grouped_bulgaria_date["confirmed"].shift()
grouped_bulgaria_date["newCases"] = grouped_bulgaria_date["newCases"].fillna(method='backfill')
grouped_bulgaria_date["newCasesRolling"] = grouped_bulgaria_date.rolling(7,1)["newCases"].mean()
grouped_bulgaria_date["confirmedLog"] = np.log(grouped_bulgaria_date["confirmed"])
#check the Trend 
ax2 = sns.lineplot(y="newCasesRolling", x="confirmedLog", data=grouped_bulgaria_date)
plt.xlabel("Total Confirmed Cases - log scale")
plt.xscale("log")
plt.ylabel("New Daily Cases")
#Other for reference
#country = "China"
country = "Italy"
grouped_ref = data[data['country'] == country].reset_index()
grouped_ref_date = grouped_ref.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

#enchance the data
grouped_ref_date["newCases"] = grouped_ref_date["confirmed"] - grouped_ref_date["confirmed"].shift()
grouped_ref_date["newCases"] = grouped_ref_date["newCases"].fillna(method='backfill')
grouped_ref_date["newCasesRolling"] = grouped_ref_date.rolling(7,1)["newCases"].mean()
grouped_ref_date["confirmedLog"] = np.log(grouped_ref_date["confirmed"])
#check the Trend 
ax3 = sns.lineplot(y="newCasesRolling", x="confirmed", data=grouped_ref_date)
plt.xlabel("Total Confirmed Cases - log scale")
plt.xscale("log")
plt.ylabel("New Daily Cases")
#Try Better plot I can send 
#import plotly.express as px
#
#df = px.data.gapminder().query("country=='Canada'")
#fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
#fig.show()

#show why this is important
exp = [2.5**i for i in range(20)]
quadratic = [0.5*i*i for i in range(20)]
i = [i for i in range(20)]
dict = {'exp': exp, 'quadratic': quadratic, 'period': i}  
TestData = pd.DataFrame(dict,index = i)


ax4 = sns.lineplot(y="quadratic", x="period", data=TestData)
plt.xlabel("Periods passed")

plt.ylabel("Quadratic - New Daily Cases")


ax4 = sns.lineplot(y="exp", x="period", data=TestData)
plt.xlabel("Periods passed")

plt.ylabel("Exponent - New Daily Cases")