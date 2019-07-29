from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import widgetbox
from bokeh.models import Slider
plot = figure()
# Add a line to the plot
plot.line(x=[1,2,3,4,5],y=[2,5,4,6,7])

# Add the plot to the current document
curdoc().add_root(plot)
#add slider
slider1 = Slider(title='slider1', start=0, end=10, step=0.1, value=2)
slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20)
layout = widgetbox(slider1,slider2)
curdoc().add_root(layout)

#Linking them and react on Change
source = ColumnDataSource(data={'x': x, 'y': y})
plot.line('x', 'y', source=source)

# Define a callback function: callback
def callback(attr, old, new):
    scale = slider.value
    new_y = np.sin(scale/x)
    source.data = {'x': x, 'y': new_y}

#
slider.on_change('value', callback)

layout = column(widgetbox(slider), plot)
curdoc().add_root(layout)3

#DROPDOWNS
from bokeh.models import ColumnDataSource, Select
# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})

# Create a new plot: plot
plot = figure()
plot.circle('x', 'y', source=source)
def update_plot(attr, old, new):
    if new == 'female_literacy': 
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }
select = Select(title="distribution", options=["female_literacy", "population"], value='female_literacy')

select.on_change('value', update_plot)
# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)

#BUTTONS
# Create a Button with label 'Update Data'
button = Button(label='Update Data')
def update():
    y = np.sin(x) + np.random.random(N)
    source.data = {'x': x, 'y': y}

button.on_click(update)

# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)


from bokeh.models import CheckboxGroup, RadioGroup, Toggle
toggle = Toggle(label='Toggle button', button_type='success')
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])
curdoc().add_root(widgetbox(toggle, checkbox, radio))


#SAMPLE APP
# Define the callback: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x'       : data.loc[yr][x],
        'y'       : data.loc[yr][y],
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to source.data
    source.data = new_data

    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Gapminder data for %d' % yr

# Create a dropdown slider widget: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)

# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)

# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)

# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)