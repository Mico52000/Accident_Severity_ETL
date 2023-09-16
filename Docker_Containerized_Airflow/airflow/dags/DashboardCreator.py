#!/usr/bin/env python
# coding: utf-8

# # Lab 11 - Creating web dashboards with Dash library in Python

# ## Reosurces / documentation 
# - [Plotly express](https://plotly.com/python/plotly-express/)
# - [Dash](https://dash.plotly.com/dash-core-components)
# - [Youtube tutorial](https://www.youtube.com/watch?v=hSPmj7mK6ng)

# In[1]:


import plotly.express as px 
from dash import Dash, dcc, html, Input, Output
import pandas as pd





def reverse_encode(df,column_name,look_up_table_df):
    df_look_table_column = look_up_table_df[look_up_table_df['column_name']==column_name]
    df_look_table_column.drop('column_name',axis =1,inplace=True)
    mapping_dict = dict()
    for index,row in df_look_table_column.iterrows():
        mapping_dict[row['Encoding']] = row['Value']
    print(df[column_name].replace(mapping_dict))
    df[column_name] = df[column_name].replace(mapping_dict)









def LineChart(df):
    class_series = df.groupby('Month').size()
    fig = px.line( x=class_series.index, y=class_series)
    fig.update_layout(xaxis_title="Month", yaxis_title="Number of Accidents" )
    return fig
def histogram(df):
    filter_col = [col for col in df if col.startswith('day_of_week')]
    dow = df[filter_col]
    dow.columns = dow.columns.str.lstrip("day_of_week_")
    dow = dow.idxmax(axis = 1)
    df['day_of_week'] =dow
    fig = px.histogram(df, x="day_of_week")
    fig.update_xaxes(categoryorder = "total descending")
    fig.update_layout(xaxis_title="Day of The Week", yaxis_title="Number of Accidents" )
    #fig.update_xaxes(tickangle=90)
    return fig
def weatherbar(df):
    filter_col = [col for col in df if col.startswith('weather_conditions')]
    dfweather = df[filter_col]
    dfweather.columns = dfweather.columns.str.lstrip("weather_conditions_")
    dfweather = dfweather.idxmax(axis = 1)
    df['weather_conditions'] =dfweather
    fig = px.histogram(df, x="weather_conditions")
    fig.update_xaxes(categoryorder = "total descending")
    fig.update_layout(xaxis_title="Weather Conditions", yaxis_title="Number of Accidents" )
    return fig
def top10districts(df):
    
    dfg = pd.DataFrame(df['local_authority_ons_district'].value_counts()[:10].sort_values(ascending=False))
    dfg.columns =['local_authority_ons_district']
    fig = px.bar(dfg,x=dfg.index,y='local_authority_ons_district')
    fig.update_xaxes(categoryorder = "total descending")
    fig.update_layout(xaxis_title="Local Authority District", yaxis_title="Number of Accidents" )
    return fig
def speedLimit(df):
    x,y =  'speed_limit','accident_severity'
    dfg = df.groupby(x)[y].value_counts(normalize=True).loc[:,"Fatal"].mul(100).rename('percent').reset_index()
    fig = px.bar(dfg,x=x,y='percent')
    fig.update_layout(xaxis_title="Speed Limit", yaxis_title="Percentage of Fatal Accidents" )
    return fig


# In[9]:


def createDashboard(filename,look_up_table_filename):
    df = pd.read_csv(filename)
    look_up_table = pd.read_csv(look_up_table_filename)
    reverse_encode(df,'local_authority_ons_district',look_up_table)
    reverse_encode(df,'accident_severity',look_up_table)
    reverse_encode(df,'speed_limit',look_up_table)
    app = Dash()
    app.layout = html.Div([
        html.H1("UK Accidents Dashboard", style={'text-align': 'center'}),
        html.Br(),
        html.H1("2011 Dataset ", style={'text-align': 'center'}),
        html.Br(),
        html.Div(),
        html.H1("Accidents Against Time", style={'text-align': 'center'}),
        dcc.Graph(figure=LineChart(df)),
        html.Br(),
        html.Div(),
        html.H1("Accidents Against Day of Week", style={'text-align': 'center'}),
        dcc.Graph(figure=histogram(df)),
        html.H1("Accidents Against Weather Conditions", style={'text-align': 'center'}),
        dcc.Graph(figure=weatherbar(df)),
        html.H1("Top 10 Most Dangerous Local Authority Districts", style={'text-align': 'center'}),
        dcc.Graph(figure=top10districts(df)),
        html.H1("Speed Limit Against Fatal Accidents Percentage", style={'text-align': 'center'}),
        dcc.Graph(figure=speedLimit(df))
    ])
    app.run_server(host = "0.0.0.0",port=8020)






