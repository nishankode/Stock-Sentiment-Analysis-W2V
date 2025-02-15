from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import pickle

class Word2Vector:

    def train_w2v(X_train):
        """ Function to Create Word2Vec """
        model = Word2Vec(X_train.tolist(), min_count=1, sg=1, vector_size=100, window=2)
        return model

    def AvgWord2Vec(model, words):
        """ Function to compute Average Word2Vec """
        vectors = [model.wv[word] for word in words if word in model.wv]

        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        else:
            return np.mean(vectors, axis=0)

    def w2v(X_train, X_test):
        """ Function to apply Average Word2Vec on dataframes """
        model = Word2Vector.train_w2v(X_train)
        X_test = X_test.apply(lambda x : Word2Vector.AvgWord2Vec(model, x))
        X_train = X_train.apply(lambda x: Word2Vector.AvgWord2Vec(model, x))

        with open('../Output/owrd2vec_model.pickle', 'wb') as f:
            pickle.dump(model, f)

        X_test = X_test.reset_index(drop=1)
        X_train = X_train.reset_index(drop=1)

        X_train = pd.DataFrame(X_train.tolist())
        X_test = pd.DataFrame(X_test.tolist())
                
        return X_train, X_test