import sys, re, pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download(['punkt', 'stopwords', 'wordnet'])

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import warnings 
warnings.filterwarnings("ignore")

stop_words = set(stopwords.words())

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('dataset', engine)
    X = df['message']
    Y = df.iloc[:, 6:]
    return X, Y, Y.columns

def tokenize(text):

    
    text = text.lower()

    text = re.sub("[^a-zA-Z0-9]", " ", text)

    text = word_tokenize(text)

    text = [word for word in text if word not in stop_words]

    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]

    return text

def build_model():

    # Create the pipeline
    pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', OneVsRestClassifier(LinearSVC()))
    ])
    
    # Create the parameters to search using GridSearchCV
    parameters = {
        'classifier__estimator__C': (0.1, 1, 10)
    }

    # Create the model using GridSearchCV
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    return model

def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)
    report = classification_report(Y_test.values, y_pred, target_names=category_names, zero_division=1)
    print(report)
    print("Model Accuracy: {}%".format(np.mean(y_pred == Y_test.values) * 100))

def save_model(model, model_filepath):

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    

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
        print('Task aborted')


if __name__ == '__main__':
    main()
