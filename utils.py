import re
import json
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import torch.nn as nn

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def read_data(file):
    file = open(file)
    data = json.load(file)
    return data

def split_data(data, labels, test_size = 0.2, random_state = 13):
    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size = test_size, random_state = random_state)
    return data_train, data_test, label_train, label_test

def preprocess(text, Tokenize = True, StopWords = True, Lemmatize = True):
    stop_words = set(stopwords.words('english'))   
    lemmatizer = WordNetLemmatizer()
    # Preprocess text
    text = text.lower()                                                       # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)   # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)                                       # Remove punctuation
    tokens = word_tokenize(text)                                              # Tokenize text  
    if StopWords:                      
        tokens = [word for word in tokens if word not in stop_words]          # Remove stopwords
    if Lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]              # Lemmatize text
    if not Tokenize:
        tokens = ' '.join(tokens)
    return tokens

def vectorize(data, method = "count", ngram_range = (1,1)):
    if method == "count":                                           # Count Vectorizer
        vectorizer = CountVectorizer(ngram_range = ngram_range)
    elif method == "tfidf":                                         # TF IDF Vectorizer                
        vectorizer = TfidfVectorizer(ngram_range = ngram_range)
    else:
        print("Invalid method")
        return
    X = vectorizer.fit_transform(data) 
    return X.toarray(), vectorizer

def train_naive_bayes(texts_vectors, labels):
    text_train, text_test, label_train, label_test = split_data(texts_vectors, labels)
    model = MultinomialNB()
    model.fit(text_train, label_train)
    predicted_labels = model.predict(text_test)
    accuracy = accuracy_score(label_test, predicted_labels)
    return model, accuracy

def PMI(word1, word2, corpus, window_size=2):
    # Initialize counts
    word1_count = 0
    word2_count = 0
    word1_word2_count = 0
    total_words = 0
    for words in corpus:
        total_words += len(words)
        for i in range(len(words)):
            if words[i] == word1:
                word1_count += 1
                for j in range(i-window_size, i+window_size+1):
                    if j >= 0 and j < len(words) and words[j] == word2:
                        word1_word2_count += 1
            if words[i] == word2:
                word2_count += 1
    # Calculate probabilities
    p_word1 = word1_count / total_words
    p_word2 = word2_count / total_words
    p_word1_word2 = word1_word2_count / total_words
    # Calculate PMI
    pmi = np.log2((p_word1_word2 / (p_word1 * p_word2)) + 0.00000001)
    return pmi