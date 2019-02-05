# import libraries
import nltk
nltk.download(['punkt','wordnet','stopwords'])
from nltk.corpus import stopwords
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sqlalchemy import create_engine
import sqlite3
from sklearn.multioutput import MultiOutputClassifier
import pickle
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import sys


def load_data(database_filepath):
    # load data from database
    conn = sqlite3.connect(database_filepath)
    sqlquery = "SELECT * FROM catmessages"
    df = pd.read_sql(sqlquery, con=conn)
    Y = df.drop(['id','message','original','genre'],axis = 1)
    X = df['message']
    labels = list(Y.columns)
    return X, Y, labels


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return (tokens)    


def build_model():
    """Define pipeline and fit model"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters =  {
    'clf__estimator__min_samples_split': [2, 5, 10,20]
    }
    cv = GridSearchCV(pipeline,param_grid = parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    cr = classification_report(Y_test, model.predict(X_test), target_names = category_names)
    print(cr)
    pass


def save_model(model, model_filepath):
    #filename = 'disaster_model_1.sav'
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()