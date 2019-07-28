rows = ['Philadelphia', 'Centre', 'Fulton']
cols = ['winner', 'Obama', 'Romney']
three_counties = election.loc[rows,cols]
#droping empty
df = titanic[['age','cabin']]
print(df.dropna(how='any').shape)
print(df.dropna(how='all').shape)
# Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh=1000, axis='columns').info())
#transforming columns

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF', 'Mean Dew PointF']].apply(lambda F: 5/9*(F-32))
# Reassign the columns df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']
print(df_celsius.head())

red_vs_blue = {'Obama':'blue' , 'Romney':'red'}

election['color'] = election['winner'].map(red_vs_blue)
#slicingm multi index
# Look up data for NY in month 1: NY_month1
NY_month1 = sales.loc[('NY', 1), :]
# Look up data for CA and TX in month 2: CA_TX_month2
CA_TX_month2 = sales.loc[(['CA', 'TX'], 2), :]
# Look up data for all states in month 2: all_month2
all_month2 = sales.loc[(slice(None), 2), :]


#Pivot tables
visitors_pivot = users.pivot(index='weekday', columns='city', values='visitors')
#Stacking and unstacking, same as group and ungroup in R, but for indexes 
print(byweekday.stack(level='weekday'))
#To swap the index
newusers.swaplevel(0, 1)

#melting
pd.melt(visitors_by_city_weekday, id_vars=['weekday'], value_name= 'visitors')

#Pivot with duplications - pivot_table
users.pivot_table(index='weekday', aggfunc='count')
#margins=True adds totals

#group by
by_class = titanic.groupby('pclass')
by_class_sub = by_class[['age','fare']]
aggregated = by_class_sub.agg(['max','median'])
print(aggregated.loc[:, ('age','max')])

# Create a groupby object: by_day
by_day = sales.groupby(sales.index.strftime('%a'))
units_sum = by_day['Units'].sum()

#transform on the whole series
standardized = gapminder_2010.groupby(['life','fertility']).transform(zscore)
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)

#Filling missing data
by_sex_class = titanic.groupby(['sex','pclass'])

def impute_median(series):
    return series.fillna(series.median())

titanic.age = by_sex_class.age.transform(impute_median)

#Filters
by_com_sum = by_company['Units'].sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum() > 35)

#Mapping the group by
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})

survived_mean_1 = titanic.groupby(under10)['survived'].mean()