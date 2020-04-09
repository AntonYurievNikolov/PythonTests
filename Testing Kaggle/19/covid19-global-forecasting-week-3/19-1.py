
import numpy as np
import pandas as pd
import math


import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

#cleaned_data.country.unique()

data_dir = Path('D:\\PythonTests\\Testing Kaggle\\19\\covid19-global-forecasting-week-4')

data = pd.read_csv(data_dir/'train.csv', parse_dates=['Date'])

data.rename(columns={'Date': 'date', 
                     'Id': 'id',
                     'Province_State':'state',
                     'Country_Region':'country',
                     'Lat':'lat',
                     'Long': 'long',
                     'ConfirmedCases': 'confirmed',
                     'Fatalities':'deaths',
                    }, inplace=True)





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
grouped_bulgaria_date["confirmedLog"] = np.log1p(grouped_bulgaria_date["confirmed"])
grouped_bulgaria_date["newCasesRollingLog"] = np.log1p(grouped_bulgaria_date["newCasesRolling"])
#check the Trend 
ax3 = sns.lineplot(y="newCasesRollingLog", x="confirmedLog", data=grouped_bulgaria_date)
plt.xlabel("Total Confirmed Cases - log scale")
plt.ylabel("New Daily Cases- log scale")

ax3 = sns.lineplot(y="newCasesRolling", x="confirmed", data=grouped_bulgaria_date)
plt.xlabel("Total Confirmed Cases")
plt.ylabel("New Daily Cases")
#Other for reference
#data.country.unique()
#country = "China"Netherlands
country = "Netherlands"
grouped_ref = data[data['country'] == country].reset_index()
grouped_ref_date = grouped_ref.groupby('date')['date', 'confirmed', 'deaths'].sum().reset_index()

#enchance the data
grouped_ref_date["newCases"] = grouped_ref_date["confirmed"] - grouped_ref_date["confirmed"].shift()
grouped_ref_date["newCases"] = grouped_ref_date["newCases"].fillna(method='backfill')
grouped_ref_date["newCasesRolling"] = grouped_ref_date.rolling(2,1)["newCases"].mean()
grouped_ref_date["confirmedLog"] = np.log1p(grouped_ref_date["confirmed"])
grouped_ref_date["newCasesRollingLog"] = np.log1p(grouped_ref_date["newCasesRolling"])
#check the Trend 
ax3 = sns.lineplot(y="newCasesRollingLog", x="confirmedLog", data=grouped_ref_date)
plt.xlabel("Total Confirmed Cases - log scale")
plt.ylabel("New Daily Cases- log scale")


ax3 = sns.lineplot(y="newCasesRolling", x="confirmed", data=grouped_ref_date)
plt.xlabel("Total Confirmed Cases")
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