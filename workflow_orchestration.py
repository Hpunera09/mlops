import pandas as pd
from prefect import task, flow
import re
from sklearn import metrics
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
nltk.download("stopwords")
nltk.download("wordnet")
len(stopwords.words("english"))

@task
def load_data(file_path):
    return pd.read_csv(file_path)
    

@task
def split_inputs_output(data, inputs, output):
    X = data[inputs]
    y = data[output]
    return X, y
    

@task
def split_train_test(X, y, test_size=0.25, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@task
def preprocess_data(X_train, X_test, y_train, y_test):
    def preprocess(raw_text):
        lemmatizer = WordNetLemmatizer()
        english_stopwords = set(stopwords.words('english'))
        negation_words = {'not', 'no', 'nor', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't","don","don't","over"}
        custom_stopwords = english_stopwords - negation_words
        
        sentence = re.sub("[^a-zA-Z]", " ", str(raw_text))
            
        sentence = sentence.lower()
        
        tokens = sentence.split()
                            
        
        clean_tokens = [t for t in tokens if t not in custom_stopwords]
            
        clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]
            
        return " ".join(clean_tokens)
    vect= TfidfVectorizer()
    X_train_vector = vect.fit_transform(X_train['Review text'].apply(preprocess))
    X_test_vector = vect.transform(X_test['Review text'].apply(preprocess))
    return X_train_vector, X_test_vector, y_train, y_test



@task
def train_model(X_train_vector, y_train, hyperparameters):
    clf = SVC(**hyperparameters)
    clf.fit(X_train_vector, y_train)
    return clf



@task
def evaluate_model(model, X_train_vector, y_train, X_test_vector, y_test):
    y_train_pred = model.predict(X_train_vector)
    y_test_pred = model.predict(X_test_vector)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score



@flow(name="SVC Training Flow")
def workflow(data_path):
    DATA_PATH = data_path
    INPUTS = ['Review text']
    OUTPUT = 'label'
    HYPERPARAMETERS =hyperparameters_svc = {'C': 1,'kernel': 'rbf'}
    
    
    df = load_data(DATA_PATH)

    
    X, y = split_inputs_output(df, INPUTS, OUTPUT)

    
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    
    X_train_vector, X_test_vector, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)

    
    model = train_model(X_train_vector, y_train, HYPERPARAMETERS)
    
   
    train_score, test_score = evaluate_model(model, X_train_vector, y_train, X_test_vector, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)


if __name__ == "__main__":
    workflow.serve(
        name="my-first-deployment",
        cron="0 0 * * 0"
    )
