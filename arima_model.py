# -*- coding: utf-8 -*-
"""
Covid Predictions App 

@author: mariano 
"""

#-------------------------------
# Import libraries
#-------------------------------

#deploy model libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

#dataframe manipulation libraries
import pandas as pd
from pandas import concat
import numpy as np
from datetime import timedelta
from matplotlib import pyplot

#arima models libraries
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


#######################################
# Define global variables
#######################################

# Define number of days considered for cumulative number
Cum_Sum_Days = 7

# Define Cum_Cases as sort name for Cum Sum
Cum_Cases = Cum_Sum_Days
    
# Definde the country 
Country = 'Spain'


#######################################
# Define functions
#######################################

#------------------------------
# function to load data 
#------------------------------


def load():
    
    url = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx"
    
    df = pd.read_excel(url,sheet_name='COVID-19-geographic-disbtributi')
    
    #------------------------------
    # Prepare data 
    #-------------------------------
    
    # Change names of some large colum names
    df = df.rename(columns={"countriesAndTerritories": "countries", 
                            "countryterritoryCode":"Country_Code",
                            "continentExp":"continent",
                            "Cumulative_number_for_14_days_of_COVID-19_cases_per_100000":"Cum_Cases_14d",
                           })
    
    df.fillna(0,inplace=True)
    
    # Transform date to the right format
    df['date'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')
    
    # Dropput the unneded columns
    df.drop('dateRep', axis=1, inplace=True)
    
    
    #--------------------------------------
    # Add rolling to all the countries
    #--------------------------------------
     
    # Sort by date all dataset to make the cumsum correctly
    df.sort_values(by = 'date', inplace=True)
    
    # Select country
    df_covid = df[df['countries'] == Country]
    
    # get the rolling sum of cases for last 14 days
    df_covid['rolling']=df_covid.cases.rolling(Cum_Sum_Days).sum()
    
    
    # Drop na
    df_covid = df_covid.dropna()
    
    return df_covid  
 

def model_covid(df_covid):
    
    #----------------------------------------
    # Prepare data for analysis 
    #----------------------------------------

    
    # Get the columns to be used 
    dataset = df_covid[['Cum_Cases_14d']]
    
    # Shift and conact to generate inputs for the timeseries problem
    # split into train and test sets
    X = dataset.values
    size = int(len(X) * 0.36)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    

    #-------------------------------
    # Function to train model
    #-------------------------------  
    
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
        
    
    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
        
        #--------------------------------------
        # Extract last value of the testing 
        #--------------------------------------
    
        # Show the last element predicted  
    Pred_last_element = predictions[len(predictions)-1]
    
        # Show the last real element  testY
    Real_last_element = history[len(history)-1]
    
        #-------------------------------------------------
        # Pred new data sample
        #-------------------------------------------------
    
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = 0
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))
    
        # Next day forecast
    Pred_Tomorrow = yhat
    
    
        #----------------------------------------------
        # Pred in n timesteps in the future
        #----------------------------------------------
    forecast_days = 14
    output = model_fit.forecast(steps=forecast_days)
    yhat = output[0]
    
    for i in range(forecast_days):
       predictions.append(yhat[i])
       
        # forectast day value
    Pred_Future = int(predictions[len(predictions)-1])    
    
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    
        
        #----------------------------------------------------------
        # Save last prediction into a file for further anlysis
        #----------------------------------------------------------
    
    # Get date of last current value
    Current_Date = df_covid.iloc[len(df_covid)-1,df_covid.columns.get_loc("date")]
    
    # Tomorrow date 
    Tomorrow_Date = Current_Date + timedelta(days=1)
    
    # Future date 
    Future_Date = Current_Date + timedelta(days=forecast_days)
    
    # Pass data to dataframe 
    data = {'Real_Point'      : [int(Real_last_element)],
            'Predicted_Point' : [int(Pred_last_element)],
            'Pred_Tomorrow'   : [int(Pred_Tomorrow)],
            'Pred_Future'     : [int(Pred_Future)],
            'Date'            : Current_Date,
            'Tomorrow_Date'   : Tomorrow_Date,
            'Future_Date'     : Future_Date}
    
    df_pred = pd.DataFrame(data, columns = ['Date',
                                            'Real_Point', 
                                            'Predicted_Point', 
                                            'Tomorrow_Date', 
                                            'Pred_Tomorrow',
                                            'Future_Date',
                                            'Pred_Future'])
    
    return df_pred


#######################################
# Create a dash website application 
#######################################

#-----------------------------
# Call the functions 
#-----------------------------

def run_covid():
    
    df_covid = load()
    
    df = model_covid(df_covid)
    
    return df


#-----------------------------
# Define app dashboard
#-----------------------------

children_text_H1 = "Covid Prediction in Country"
subtitle = "Covid 19 prediction using ARIMA. Prediction over cumulative cases."
title1 = "Covid 19 prediction for last updated day:"
title2 = "Covid 19 prediction for day"



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    
    html.H1(id='H1',children=children_text_H1),

    html.Div(children= 
        subtitle
    ),
    
    html.Hr(),
    
    html.Button('Click to train', id='button'),
    html.H3(id='button-clicks'),

    html.Hr(),
    
   # html.Div(id='output'),
    
    html.Hr(),
   
    dcc.Graph(id='graph1'),
    
    
    dcc.Graph(id='graph2'),
    
    dcc.Graph(id='graph3'),
   


])


# =============================================================================
# @app.callback(
#     Output('graph1', 'figure'),
#     [Input('button', 'n_clicks')])
# def clicks(n_clicks):
#     
#     fig = plotly
#     
#     fig.append_trace({
#         'x': data['time'],
#         'y': data['Altitude']})
#     
#     
#     return fig
# =============================================================================

@app.callback([Output('graph1', 'figure'),
              Output('graph2', 'figure'),
              Output('graph3', 'figure'),
              Output('H1', 'children')],
              [Input('button', 'n_clicks')])

def update_figure(n_clicks):
    
    df = run_covid()
    
 #   df_covid = load()
    
  #  df = model_covid(df_covid)
     
# =============================================================================
#      fig = {
#                     'data': [
#                       {'x': [1], 'y': [[5]], 'type': 'bar', 'name': 'Prediction'},
#                       {'x': [2], 'y': [[2]], 'type': 'bar', 'name': u'Real'},
#                       ]}
#     
# =============================================================================
     
    return  {
                    'data': [
                       {'x': [1], 'y': [df.Predicted_Point[0]], 'type': 'bar', 'name': 
                        'Prediction'},
                       {'x': [2], 'y': [df.Real_Point[0]], 'type': 'bar', 'name': 
                        u'Real'},
                      ],
                    'layout': {
                      'title': "Covid 19 prediction for last updated day: {}".format(df.Date[0]) 
                    }
            } ,  {
                    'data': [
                        {'x': [1], 'y': [df.Pred_Tomorrow[0]], 'type': 'bar', 'name': 
                         'Prediction_Tomorrow'}
                      ],
                    'layout': {
                      'title': "Covid 19 prediction for day: {}".format(df.Tomorrow_Date[0])
                    }
            } , {
                    'data': [
                        {'x': [1], 'y': [df.Pred_Future[0]], 'type': 'bar', 'name': 
                         'Prediction_Future'}
                      ],
                    'layout': {
                      'title': "Expected value in 14 days. Date: {}".format(df.Future_Date[0])
                    }
            }, "Covid Prediction in Country: {}".format(Country)  
                




if __name__ == '__main__':
    app.run_server(debug=False,use_reloader=False, threaded=False)
    
    
    
# =============================================================================
#  '''
#     dcc.Graph(
#         id='graph1',
#         figure={
#             'data': [
#                 {'x': [1], 'y': [df.Predicted_Point[0]], 'type': 'bar', 'name': 'Prediction'},
#                 {'x': [2], 'y': [df.Real_Point[0]], 'type': 'bar', 'name': u'Real'},
#             ],
#             'layout': {
#                 'title': title1 
#             }
#         }
#     ),
# 
#     dcc.Graph(
#         id='graph2',
#         figure={
#             'data': [
#                 {'x': [1], 'y': [df.Pred_Tomorrow[0]], 'type': 'bar', 'name': 'Prediction_Tomorrow'}
#             ],
#             'layout': {
#                 'title': title2 
#             }
#         }
#     ),'''
# =============================================================================

