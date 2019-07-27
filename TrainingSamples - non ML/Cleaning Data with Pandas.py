import pandas as pd
import matplotlib.pyplot as plt
# Create the boxplot
df.boxplot(column='initial_cost', by='Borough', rot=90)
plt.show()
#unpivot with melt, pivot 
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'])
airquality_pivot = airquality_dup.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading', aggfunc=np.mean)

#spliting
# Create the 'str_split' column
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

# Create the 'type' column
ebola_melt['type'] = ebola_melt.str_split.str.get(0)

# Create the 'country' column
ebola_melt['country'] = ebola_melt.str_split.str.get(1)

#merging data
# concatenating everything in a directory
import glob
import pandas as pd

pattern = '*.csv'
csv_files = glob.glob(pattern)
frames = []
for csv in csv_files:
    df = pd.read_csv(csv)
    frames.append(df)
    
uber = pd.concat(frames)

m2o = pd.merge(left=site, right=visited, left_on='name', right_on='site')

#converting data
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')

tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])
#missing values
tracks_no_duplicates = tracks.drop_duplicates()

oz_mean = airquality.Ozone.mean()
airquality['Ozone'] = airquality.Ozone.fillna(oz_mean)