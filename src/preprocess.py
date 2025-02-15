import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

class Preprocess:

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
        data['News'] = data['News'].apply(lambda x : Preprocess.stop_lemma(x, stop_word_dict, lemmatizer))

        # Performing Target Mapping
        data['Sentiment'] = data['Sentiment'].map({'neutral' : 0, 'positive' : 1, 'negative' : -1})

        # Train test splitting
        X_train, X_test, y_train, y_test = train_test_split(data['News'], data['Sentiment'], test_size=0.2, stratify=data['Sentiment'], random_state=42)
        
        y_train = y_train.to_frame()
        y_test = y_test.to_frame()
        
        return X_train, X_test, y_train, y_test