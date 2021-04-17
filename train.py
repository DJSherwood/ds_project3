# ML Pipline Preparation
# import libraries
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidTransformer

# load data from database
engine = create_engine('sqlite:///Messages.db')
df = pd.read_sql_query("SELECT * from Messages", con=engine)

# Set variables
X = df[['message']]
X = X.message.tolist()
y = df.drop(columns=['id','message','original','genre'])

# write a tokenization function to process data
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def toenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and reomve stop words
    tokens = [lemmatizer.lemmatize(word) for wors in tokens if word not in stop_words]
    
    return tokens

# build pipeline
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from skelarn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

pipeline = Pipeline([
    ('vectorize', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer(smooth_idf=False)),
    ('classify', MultiOutputClassifier(estimator=RandomForestClassifier())
])

# train pipeline
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# train model
rf = pipeline.fit(X=X_train, y=y_train)

