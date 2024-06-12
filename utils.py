import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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
    text_train, text_test, label_train, label_test = train_test_split(texts_vectors, labels, test_size = 0.2, random_state = 42)
    model = MultinomialNB()
    model.fit(text_train, label_train)
    predicted_labels = model.predict(text_test)
    accuracy = accuracy_score(label_test, predicted_labels)
    return model, accuracy


