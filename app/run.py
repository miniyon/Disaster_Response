
import json
import plotly
import pandas as pd
import sys,os
import sqlite3
import pickle
import numpy as np
path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+ r'\models'
sys.path.insert(0,path)

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from train_classifier import TextClean


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
# engine = create_engine('sqlite:///../data/YourDatabaseName.db')
conn = sqlite3.connect('../data/DisasterResponse.db')
df = pd.read_sql('''select * from disaster_response_tweets''',con=conn)

# load model
model = pickle.load(open("../models/classifier.pkl",'rb'))



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    reverse_genre_dict = {0:'news',1:'direct',2:'social'}
    df['genre'] = df['genre'].apply(lambda x : reverse_genre_dict[x])
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    df_sub = df[df.columns[3:]].astype(int)
    df_sub = df_sub.sum().reset_index()
    df_sub.rename(columns = {'index':'Category',0:'Count'},inplace=True)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_sub['Category'].unique().tolist(),
                    y=df_sub['Count'].values.tolist()
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict(query)[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))
    
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():

    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()