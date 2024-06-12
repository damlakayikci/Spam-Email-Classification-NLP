import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def read_data(file):
    file = open(file)
    data = json.load(file)
    return data

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
        preprocessed_text = ' '.join(tokens)
    return preprocessed_text


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