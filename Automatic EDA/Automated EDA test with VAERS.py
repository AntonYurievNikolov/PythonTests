import pandas as pd
from pandas_profiling import ProfileReport
df=pd.read_csv('2020VAERSDATA.csv')
df

design_report = ProfileReport(df)
design_report.to_file(output_file='report.html')


#sweetviz
import sweetviz as sv
sweet_report = sv.analyze(df.DIED)
sweet_report.show_html('sweetviz_report.html')

df1 = sv.compare(df[210:], df[:90])
df1.show_html('sweetvizCompare2.html')

#autoviz
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
df = AV.AutoViz('2020VAERSDATA.csv')

df = AV.AutoViz('2020VAERSDATA.csv', depVar='DIED')