import utils

file  = "Oppositional_thinking_analysis_dataset.json"

if __name__ == "__main__":

    data = utils.read_data(file)

    # Get text data and labels
    text_data = [data[i]['text'] for i in range(len(data))]
    label_mapping = {'CONSPIRACY': 1, 'CRITICAL': 0}
    labels = [label_mapping[data[i]['category']] for i in range(len(data))]

    # Preprocess text data
    # 1 without stop words and lemmatized
    preprocessed_l = [utils.preprocess(text, False) for text in text_data]
    # 2 without stop words and not lemmatized
    preprocessed = [utils.preprocess(text, False, True, False) for text in text_data]
    # 3 with stop words and lemmatized
    preprocessed_swl = [utils.preprocess(text, False, False, True) for text in text_data]
    
    # Vectorize text data
    # Bag of Words Vectorizer
    BoW_vector_l, vectorizer = utils.vectorize(preprocessed_l)
    BoW_vector, vectorizer = utils.vectorize(preprocessed)
    BoW_vector_swl, vectorizer = utils.vectorize(preprocessed_swl)
    # N-gram Vectorizer
    ngram_range = (2,2)
    N_gram_vector_l, vectorizer = utils.vectorize(preprocessed_l, "count", ngram_range)
    N_gram_vector, vectorizer = utils.vectorize(preprocessed, "count", ngram_range)
    N_gram_vector_swl, vectorizer = utils.vectorize(preprocessed_swl, "count", ngram_range)
    # TF-IDF Vectorizer
    TF_IDF_vector_l, vectorizer = utils.vectorize(preprocessed_l, "tfidf")
    TF_IDF_vector, vectorizer = utils.vectorize(preprocessed, "tfidf")
    TF_IDF_vector_swl, vectorizer = utils.vectorize(preprocessed_swl, "tfidf")

    # Train Naive Bayes model
    _ , accuracy_BoW_l = utils.train_naive_bayes(BoW_vector_l, labels)
    _ , accuracy_BoW = utils.train_naive_bayes(BoW_vector, labels)
    _ , accuracy_BoW_swl = utils.train_naive_bayes(BoW_vector_swl, labels)

    _ , accuracy_N_gram_l = utils.train_naive_bayes(N_gram_vector_l, labels)
    _ , accuracy_N_gram = utils.train_naive_bayes(N_gram_vector, labels)
    _ , accuracy_N_gram_swl = utils.train_naive_bayes(N_gram_vector_swl, labels)

    _ , accuracy_TF_IDF_l = utils.train_naive_bayes(TF_IDF_vector_l, labels)
    _ , accuracy_TF_IDF = utils.train_naive_bayes(TF_IDF_vector, labels)
    _ , accuracy_TF_IDF_swl = utils.train_naive_bayes(TF_IDF_vector_swl, labels)

    print("Accuracy of BoW       L:", accuracy_BoW_l)
    print("Accuracy of BoW        :", accuracy_BoW)
    print("Accuracy of BoW     SWL:", accuracy_BoW_swl)

    print("Accuracy of N-gram    L:", accuracy_N_gram_l)
    print("Accuracy of N-gram     :", accuracy_N_gram)
    print("Accuracy of N-gram  SWL:", accuracy_N_gram_swl)

    print("Accuracy of TF-IDF    L:", accuracy_TF_IDF_l)
    print("Accuracy of TF-IDF     :", accuracy_TF_IDF)
    print("Accuracy of TF-IDF  SWL:", accuracy_TF_IDF_swl)
    