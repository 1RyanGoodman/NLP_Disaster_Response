import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


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
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('catmessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    top_concern_counts = df.drop(['id','message','original','genre'],axis = 1).sum().sort_values(ascending = False).head(11)[1:]
    top_concern_names = list(top_concern_counts.index)
    bot_concern_counts = df.drop(['id','message','original','genre'],axis = 1).sum().sort_values(ascending = True).head(10)
    bot_concern_names = list(bot_concern_counts.index)    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    #x=genre_names,
                    #y=genre_counts
                    x=top_concern_names,
                    y=top_concern_counts                    
                )
            ],

            'layout': {
                'title': 'Top Ten Message Concerns',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Concern"
                }
            }
        },
        {
            'data': [
                Bar(
                    #x=genre_names,
                    #y=genre_counts
                    x=bot_concern_names,
                    y=bot_concern_counts                    
                )
            ],

            'layout': {
                'title': 'Lowest Ten (Relative) Message Concerns',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Concern"
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
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

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