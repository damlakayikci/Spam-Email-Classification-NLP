import utils
import FFNN
import sys
import pickle
import random 

file  = "Oppositional_thinking_analysis_dataset.json"

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide the model name as an argument.")
        sys.exit(0)

    operation =  sys.argv[1]

    # Read data
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

    if operation == "nb":
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
        sys.exit(0)
    if operation == "ffnn":
        # Split data into training and testing
        x_train, x_test, y_train, y_test = utils.split_data(BoW_vector, labels)
        epochs = 30
        # Train Feed Forward Neural Network
        # BoW vector
        # if model.pkl exists, load it
        try:
            with open('models/model_bow.pkl', 'rb') as file:
                ffnn_bow = pickle.load(file)
        except:
            ffnn_bow = FFNN.FFNN(x_train.shape[1], 10, 1)
            ffnn_bow.train(x_train, y_train, 0.1, epochs)
            # Save model
            with open('models/model_bow.pkl', 'wb') as file:
                pickle.dump(ffnn_bow, file)
        
        # N-gram vector
        # if model.pkl exists, load it
        try:
            with open('models/model_ngram.pkl', 'rb') as file:
                ffnn_ngram = pickle.load(file)
        except:
            ffnn_ngram = FFNN.FFNN(x_train.shape[1], 10, 1)
            ffnn_ngram.train(x_train, y_train, 0.1, epochs)
            # Save model
            with open('models/model_ngram.pkl', 'wb') as file:
                pickle.dump(ffnn_ngram, file)
        
        # TF-IDF vector
        # if model.pkl exists, load it
        try:
            with open('models/model_tfidf.pkl', 'rb') as file:
                ffnn_tfidf = pickle.load(file)
        except:
            ffnn_tfidf = FFNN.FFNN(x_train.shape[1], 10, 1)
            ffnn_tfidf.train(x_train, y_train, 0.1, epochs)
            # Save model
            with open('models/model_tfidf.pkl', 'wb') as file:
                pickle.dump(ffnn_tfidf, file)
        
        # Predict
        err = ffnn_bow.predict(x_test, y_test)
        print("Error of BoW:", err)
        err = ffnn_ngram.predict(x_test, y_test)
        print("Error of N-gram:", err)
        err = ffnn_tfidf.predict(x_test, y_test)
        print("Error of TF-IDF:", err)


    if operation == "stats":
        # number of unique words

        pass

    if operation == "pmi":
        print("Computing PMI...")
        # Preprocess text data
        preprocessed_pmi = [utils.preprocess(text, True, False, True) for text in text_data]
        # choose random words
        words=[]
        for i in range(10):
            words.append(random.choice(preprocessed_pmi[random.randint(0, len(preprocessed_pmi) - 1)]))
        print(words)
        PMIs = []
        # Compute PMI
        for i in range(len(words)):
            pmis = {}
            for j in range(len(words)):
                if i == j:
                    continue
                PMI = utils.PMI(words[i], words[j], preprocessed_pmi)
                print(f"PMI of {words[i]} and {words[j]}\t: {PMI}")
                pmis[words[j]] = PMI
            PMIs.append(pmis)

        def sort_dict_by_value(data):
            sorted_keys_desc = [item[0] for item in sorted(data.items(), key=lambda item: item[1], reverse=True)]
            return sorted_keys_desc
        # Print most similar words for each word
        for i in range(len(words)):
            word_dict = PMIs[i]
            print(f"Most similar words to \'{words[i].upper()}\'", *sort_dict_by_value(word_dict)[:3], sep=", ")
        
        
        sys.exit(0)


    