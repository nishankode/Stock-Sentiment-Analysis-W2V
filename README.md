# Sentiment Analysis with Word2Vec & Logistic Regression

This repository contains a Pipeline that implements a sentiment analysis pipeline on news data. The project leverages natural language processing (NLP) techniques and machine learning to classify news articles into three sentiment categories: negative (-1), neutral (0), and positive (1).

## Overview

The pipeline consists of the following main stages:

1. **Data Loading:**  
   Reads a CSV file containing news articles and sentiment labels.

2. **Preprocessing:**  
   - **Tokenization & Lowercasing:** Cleans the text by removing special characters and converting all words to lowercase.  
   - **Stopword Removal & Lemmatization:** Eliminates common stopwords and reduces words to their base form using NLTK.

3. **Data Splitting:**  
   Splits the processed dataset into training and testing subsets with stratified sampling.

4. **Word Embedding Generation:**  
   Trains a Word2Vec model using the training data and creates averaged word embeddings for each news article.

5. **Model Training & Evaluation:**  
   - **Baseline Model:** A Logistic Regression classifier is trained using the averaged embeddings.  
   - **Performance Metrics:** The classifier is evaluated using precision, recall, and f1-score metrics.


## Setup & Requirements

Ensure you have the following Python packages installed:

- `pandas`
- `numpy`
- `nltk`
- `gensim`
- `scikit-learn`

You can install these dependencies using pip:

```bash
pip install pandas numpy nltk gensim scikit-learn
```

In addition, download the required NLTK data (if not already available):

```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```