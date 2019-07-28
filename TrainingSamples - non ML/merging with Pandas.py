#reading multiple files/streams
import pandas as pd
filenames = ['Gold.csv', 'Silver.csv', 'Bronze.csv']
dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(filename))
#Reindex
names_1981.reindex(names_1881.index)
#%change
yearly = post2008.resample('A').last()
# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change() * 100

#Combining datasets with multi index
for medal in medal_types:
    file_name = "%s_top5.csv" % medal
    medal_df = pd.read_csv(file_name, index_col='Country')
    medals.append(medal_df)
# Concatenate medals: medals
medals = pd.concat(medals, keys=['bronze', 'silver', 'gold'])
#then slice the index 
idx = pd.IndexSlice
print(sales.loc[idx[:, 'Mediacore'], :])
#inner joins
medal_list = [bronze, silver, gold]
medals = pd.concat(medal_list, keys=['bronze', 'silver', 'gold'], axis=1, join='inner')
#merging
merge_by_id = pd.merge(revenue,managers,on='branch_id')
#inner join
combined = pd.merge(revenue,managers,left_on='city',right_on='branch')
#right join
 pd.merge(sales, managers, left_on=['city','state'], right_on=['branch','state'], how='left')
 #this is default to outer join
 tx_weather_ffill = pd.merge_ordered(austin , houston,on='date',suffixes=['_aus','_hus'],fill_method='ffill')
 
 #case study
 
 # Import pandas
import pandas as pd

# Create empty dictionary: medals_dict
medals_dict = {}
for year in editions['Edition']:
    file_path = 'summer_{:d}.csv'.format(year)
    medals_dict[year] = pd.read_csv(file_path)
    medals_dict[year] = medals_dict[year][['Athlete', 'NOC', 'Medal']]
    medals_dict[year]['Edition'] = year
    
# Concatenate medals_dict: medals
medals = pd.concat(medals_dict, ignore_index=True)
medals.pivot_table(index='Edition', values='Athlete', columns='NOC', aggfunc='count')
totals = editions.set_index('Edition')
totals = totals['Grand Total']
# Divide medal_counts by totals: fractions
fractions = medal_counts.divide(totals,axis=0)
#!!! CHECK THIS BELOW FURTHER
# Apply the expanding mean: mean_fractions
mean_fractions = fractions.expanding().mean()

fractions_change = mean_fractions.pct_change()*100