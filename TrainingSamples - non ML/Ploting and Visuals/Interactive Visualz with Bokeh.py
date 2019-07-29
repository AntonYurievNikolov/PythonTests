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

#Data Sources
from bokeh.plotting import ColumnDataSource
source = ColumnDataSource(df)
# Add circle glyphs to the figure p
p.circle(x='Year', y='Time', color='color', size=8, source=source)
output_file('sprint.html')
show(p)

#Hover and Select
# Create a figure with the "box_select" tool: p
p = figure(x_axis_label='Year', y_axis_label='Time',tools='box_select')
p.circle(x='Year', y='Time', selection_color='red', nonselection_alpha=0.1, source=source)

#hover
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')
hover = HoverTool(tooltips=None, mode='vline')
#with tips
hover = HoverTool(tooltips=[ ('Country', '@Country')])
# Add the hover tool to the figure p 
p.add_tools(hover)

#Color mapping
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')


#LAYOUTS
from bokeh.layouts  import row

# Import row from bokeh.layouts
from bokeh.layouts import row,column

p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
p1.circle('fertility', 'female_literacy', source=source)
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')
p2.circle('population', 'female_literacy', source=source)
layout = row(p1, p2)
layout = column(p1, p2)
output_file('fert_row.html')
show(layout)

#Grids
from bokeh.layouts import gridplot

row1 = [p1 , p2]
row2 = [p3 , p4]
layout = gridplot([row1,row2])

#Tabs
# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel
tab1 = Panel(child=p1, title='Latin America')
tab2 = Panel(child=p2, title='Africa')
tab3 = Panel(child=p3, title='Asia')
tab4 = Panel(child=p4, title='Europe')

from bokeh.models.widgets import Tabs
layout = Tabs(tabs=[tab1, tab2, tab3,  tab4])

#Linking Plots
p2.x_range = p1.x_range
p2.y_range = p1.y_range
p3.x_range = p1.x_range
p4.y_range = p1.y_range
output_file('linked_range.html')
show(layout)

#Selections will be linked if we use the same DataSource

#Legends
#legend="Africa" to the glyph
p.legend.location = 'bottom_left'
p.legend.background_fill_color = 'lightgray'

