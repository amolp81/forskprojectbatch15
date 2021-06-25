#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 22:06:25 2021

@author: amol

It has 3 sections, 
section 1 shows the pie chart based on reviews (+/-ve)
section 2 shows dropdown of 1000 reviews and based on selected review shows (+/-ve)
section 3 shows textarea for entering review and  based on review shows (+/-ve)

"""

# Importing the libraries
import pickle
import pandas as pd
import webbrowser
# !pip install dash
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output , State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot  as plt

import os



# Declaring Global variables
project_name = None
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions = True)

# global scrappedReviews
# scrappedReviews = pd.read_csv('scrappedReviews.csv')


# Defining My Functions
def load_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    
    global df
    df = pd.DataFrame()
  
    global pickle_model
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = pickle.load(file)

    global vocab
    file = open("feature.pkl", 'rb') 
    vocab = pickle.load(file)
    
    PATH = 'assets/piechart_reviews2.jpg'
    
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print('file exists ')
    else:
    
        transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)


        positivity = []
        for i in range(0,len(scrappedReviews['reviews'])):
            vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([scrappedReviews['reviews'][i]]))
            #positivity[i] = pickle_model.predict(vectorised_review)
            positivity.append( pickle_model.predict(vectorised_review)[0])
        
        df['Positivity'] =   positivity
    
        global Positivity
    
        Positivity = ['negative reviews', 'positive reviews']

        plt.title('pie-chart')
        plt.pie(df['Positivity'].value_counts().sort_index(),  autopct='%.2f%%')

        plt.axis('equal')
        plt.legend(labels=Positivity,loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))
    
        plt.savefig('assets/piechart_reviews2.jpg', dpi=300, bbox_inches='tight')
    

def check_review(reviewText):

    #reviewText has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model preidctor
    #load it later

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))


    # Add code to test the sentiment of using both the model
    # 0 == negative   1 == positive

    
    return pickle_model.predict(vectorised_review)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
def create_app_ui():
    
    colors = {
   'background': 'slategrey',
   'text': '#ff0033'
   }
    
    main_layout = html.Div(style={'backgroundColor': colors['background']}, children=
    [
    html.H1(id='Main_title', children = "Sentiment Analysis with Insights",style={'text-align':'center','color': 'purple'}),
    
    #pie-chart display 
    html.Center(html.Img(src='assets/piechart_reviews2.jpg',style={"vertical-align":"middle",'height':'400px'},title='pie-chart')),
    
    html.Br(children = None),
    
    html.Center(html.Img(src='assets/word_cloud.jpg',style={"vertical-align":"middle",'height':'500px'},title='word-cloud')),
   
    html.Br(children = None),
    html.Br(children = None),
    
    #dropdown display and accordingly selected review displays Positive or negative
    dcc.Dropdown(id='dropdown', options=[
        {'label': i, 'value': i} for i in scrappedReviews['reviews'].sample(1000).to_numpy()
    ], style={'color': 'blue'}, multi=False, optionHeight=80),
    
    
    html.H1(children = None, id='result',style={'text-align':'center','color': 'darkgreen'}),
    
    html.H1(children = None, id='result1',style={'text-align':'center','color': 'red'}),
    
    html.Br(children = None),
     
    #textarea for entering review and based on that displays Positive or negative
    dcc.Textarea(
        id = 'textarea_review',
        placeholder = 'Enter the review here.....',
        style = {'width':'100%', 'height':40}
        ),
    
    dbc.Button(
        children = 'FInd Review',
        id = 'button_review',
        color = 'primary',
        style= {'width':'100%'}
        ),
    
     html.H1(children = None, id='result2',style={'text-align':'center','color': 'darkgreen'}),
     html.H1(children = None, id='result3',style={'text-align':'center','color': 'red'}),
    ]    
    )
    
    return main_layout



'''
Event Handling 
When some clicks the button call my method update_app_ui

Wiring 
Object      Event    Function 
Button      Click    update_app_ui

Decorators and callbacks mechanism is a way to implment wiring in python
Input  === Arguments to your callback
Output === return of your callback 

'''

'''
@app.callback(
    Output( 'result'   , 'children'     ),
    [
    Input( 'textarea_review'    ,  'value'    )
    ]
    )
def update_app_ui(textarea_value):
    
    print("Data Type = ", str(type(textarea_value)))
    print("Value = ", str(textarea_value))

    response = check_review(textarea_value)

    if (response[0] == 0):
        result = 'Negative'
    elif (response[0] == 1 ):
        result = 'Positive'
    else:
        result = 'Unknown'

    return result
'''

#dropdown display and accordingly selected review displays Positive or negative
@app.callback(
    [
    Output( 'result'   , 'children'     ),
    Output( 'result1'   , 'children'     ),
    ],
    [
    Input( 'dropdown', 'value'   )
    ]
    )
def update_app_ui_2(dropdown_value):

    print("Data Type = ", str(type(dropdown_value)))
    print("Value = ", str(dropdown_value))
    


    if dropdown_value:
       
        text = dropdown_value
        
        
        response = check_review(text)
        
        #print('response is ',response[0])
        
        if (response[0] == 0):
            result1 = 'Negative'
            result = ""
        elif (response[0] == 1 ):
            result = 'Positive'
            result1 = ""
        else:
            result = 'Unknown'
            result1 = 'Unknown'
            
        return result,result1
    else:
        return "",""

#textarea display and accordingly entered review displays Positive or negative    
@app.callback(
    [
    Output( 'result2'   , 'children'     ),
    Output( 'result3'   , 'children'     ),
    ],
    [
    Input( 'button_review', 'n_clicks'   )
    ],
    [
    State( 'textarea_review'  ,   'value'  )
    ]
    )
def update_app_ui(n_clicks, textarea_value):

    print("Data Type = ", str(type(n_clicks)))
    print("Value = ", str(n_clicks))


    print("Data Type = ", str(type(textarea_value)))
    print("Value = ", str(textarea_value))
    


    # print("Data Type = ", str(type(textarea_value)))
    # print("Value = ", str(textarea_value))

    if (n_clicks > 0):
        
        response = check_review(textarea_value)
        if (response[0] == 0):
            result3 = 'Negative'
            result2 = ""
        elif (response[0] == 1 ):
            result2 = 'Positive'
            result3 = ""
        else:
            result2 = 'Unknown'
            result3 = 'Unknown'
        
        return result2,result3
        
    else:
        return "",""
    
    

# Main Function to control the Flow of your Project
def main():
    print("Start of your project")
    load_model()
    open_browser()
    #update_app_ui()
    
    
    global scrappedReviews
    global project_name
    global app
    
    project_name = "Sentiment Analysis with Insights"
    #print("My project name = ", project_name)
    #print('my scrapped data = ', scrappedReviews.sample(5) )
    
    # favicon  == 16x16 icon ----> favicon.ico  ----> assests
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()
    
    
    
    print("End of my project")
    project_name = None
    scrappedReviews = None
    app = None
    
        
# Calling the main function 
if __name__ == '__main__':
    main()
    
    
    