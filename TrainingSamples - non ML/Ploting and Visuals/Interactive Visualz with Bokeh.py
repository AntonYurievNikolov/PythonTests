from bokeh.plotting import figure
from bokeh.io import output_file, show
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')
p.circle(fertility_latinamerica, female_literacy_latinamerica, alpha=0.8, size=10, color='blue')
p.x(fertility_africa , female_literacy_africa)
output_file('fert_lit_separate.html')
show(p)

#patches
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')
p.circle(date, price, fill_color='white', size=4)
p.line(date,price)
# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]
# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats,  ut_lats]
p.patches(x,y,line_color='white')
