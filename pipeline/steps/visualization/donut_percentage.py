"""
import plotly.graph_objects as go

percentage = 0.8
labels = ['visible', 'non-visible']
values = [percentage, 1-percentage]

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.9)])
fig.update_layout(
    # Add annotations in the center of the donut pies.
    annotations=[dict(text="{}%".format(percentage), x=0.18, y=0.5, font_size=20, showarrow=False)])
fig.show(renderer="png")
"""

import plotly.express as px
import pandas as pd
import plotly.io as pio
pio.renderers.default = 'png'

percentage = 0.8

df = pd.DataFrame({'Percentage': [percentage, 1 - percentage]})
fig = px.pie(df, values='Percentage', names=None, title='Percentage Donut', hole=0.9)
fig.show()
