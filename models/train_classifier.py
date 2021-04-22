import sys

from sqlalchemy import create_engine
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    # load engine
    engine = create_engine('sqlite:///' + str(database_filepath))
    # load dataset
    df = pd.read_sql_query("SELECT * from DisasterResponse", con=engine)
    # create X,Y
    X = df[['message']]
    X = X.message.tolist()
    Y = df.drop(columns=['id','message','original','genre'])
    # create category name variable
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('classify', MultiOutputClassifier(RandomForestClassifier(max_depth=5)))
    ])

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    # generate predictions
    Y_pred = model.predict(X_test)
    # find the number of columns in prediction
    _, ncols = Y_pred.shape
    # convert pandas dataframe to a numpy array
    Y_test = Y_test.values
    # print classification report - category names is supposed to be in here somehow
    for i in range(0,ncols):
        print(category_names[i])
        print('==========')
        print(classification_report(Y_test[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        fitted_model = model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(fitted_model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(fitted_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()