import sys
import nltk
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
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    #messages = pd.read_csv('messages.csv')
    #categories = pd.read_csv('categories.csv')
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #### Dropping duplicate Ids
    print(categories.shape)
    categories.drop_duplicates('id',inplace = True)
    print(categories.shape)
    print(messages.shape)
    messages.drop_duplicates('id',inplace = True)
    print(messages.shape)
    # merge datasets
    df = pd.merge(messages,categories,left_on = 'id',right_on = 'id', how = 'inner')
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    #categories = categories['categories'].str.split(';',expand = True)
    categories = df['categories'].str.split(';',expand = True)
    # resetting index, concatenation has some problems otherwise, and the records/rows don't line up correctly
    categories.reset_index(inplace=True)
    categories = categories.drop('index',axis = 1)
    # select the first row of the categories dataframe
    row = categories[:1].transpose()[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # for some reason there was a 2 in there, which messed up my ML Pipeline later on, so fixing that here now
    categories = categories.replace(2,1)
    # drop the original categories column from `df`
    df.drop(['categories'],axis = 1,inplace = True)
    df = pd.concat([df,categories],axis = 1)
    return df


def save_data(df, database_filename):
    #Create enginer
    #engine = create_engine('sqlite:///catmessages.db')
    engine = create_engine('sqlite:///' + database_filename)
    #Delete if rewriting
    #engine.execute('DELETE from "catmessages"')
    engine.execute('DROP TABLE IF EXISTS catmessages')
    #Write to Database
    df.to_sql('catmessages', engine, index=False, if_exists = 'append')
    #Check Table Shape - should be 26180
    import sqlite3
    #conn = sqlite3.connect('DisasterResponse.db')
    conn = sqlite3.connect(database_filename)
    sqlquery = "SELECT * FROM catmessages"
    df = pd.read_sql(sqlquery, con=conn)
    print('Shape of data should be 26180, 40 and it is {}'.format(df.shape))
    pass  

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        print('Cleaned data saved to database!')
        #print('Shape of data should be 26180 and it is {df.shape[0]}')
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
        
if __name__ == '__main__':
    main()

#import sys
#def load_data(messages_filepath, categories_filepath):
#    pass
#def clean_data(df):
#    pass
#def save_data(df, database_filename):
#    pass  
#def main():
#    if len(sys.argv) == 4:
#
#        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
#
#        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
#              .format(messages_filepath, categories_filepath))
#        df = load_data(messages_filepath, categories_filepath)
#
#        print('Cleaning data...')
#        df = clean_data(df)
#        
#        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
#        save_data(df, database_filepath)
#        
#        print('Cleaned data saved to database!')
#    
#    else:
#        print('Please provide the filepaths of the messages and categories '\
#              'datasets as the first and second argument respectively, as '\
#              'well as the filepath of the database to save the cleaned data '\
#              'to as the third argument. \n\nExample: python process_data.py '\
#              'disaster_messages.csv disaster_categories.csv '\
#              'DisasterResponse.db')
#
#
#if __name__ == '__main__':
#    main()
#    
    
   