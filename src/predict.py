import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

class MakePredicitons:

    def stop_lemma(x, stop_word_dict, lemmatizer):
        """ Function to Perform Stopwords Removal and Lemmatization"""
        result = [lemmatizer.lemmatize(word) for word in x if word not in stop_word_dict]
        return result

    def preprocess_data(data):
        
        "Function to take dataframe as inpt, perform preprocessing and return splitted dataframes."

        # Getting rid of non-alphanumeric characters
        data['News'] = data['News'].apply(lambda x : re.sub(r'[^a-zA-Z0-9]', ' ', x))

        # Lowercasing for consistency
        data['News'] = data['News'].str.lower()

        # Tokenizing the sentences to words
        data['News'] = data['News'].apply(lambda x : word_tokenize(x))

        # Performing Stopword Removal and Lemmatization
        stop_word_dict = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        data['News'] = data['News'].apply(lambda x : MakePredicitons.stop_lemma(x, stop_word_dict, lemmatizer))

        return data
    
    def AvgWord2Vec(model, words):
        
        vectors = [model.wv[word] for word in words if word in model.wv]

        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        else:
            return np.mean(vectors, axis=0)


    def vectorize(data):
            
        with open('../Output/owrd2vec_model.pickle', 'rb') as f:
            w2v_model = pickle.load(f)

        data['News'] = data['News'].apply(lambda x: MakePredicitons.AvgWord2Vec(w2v_model, x))

        return data['News']

    def get_predictions(data):

        data = pd.DataFrame(data.tolist())

        with open('../Output/best_model.pickle', 'rb') as f:
            best_model = pickle.load(f)

        predictions = best_model.predict(data)

        return predictions
    
    def process(data):
        data_cp = data.copy()
        data = MakePredicitons.preprocess_data(data)
        data = MakePredicitons.vectorize(data)
        predictions = MakePredicitons.get_predictions(data)
        data_cp['Predictions'] = predictions
        data_cp['Predictions'] = data_cp['Predictions'].map({0: 'neutral', -1: 'negative', 1: 'positive'})

        return data_cp
