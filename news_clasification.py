import sys
from datetime import datetime
import pickle
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import urllib3
import xmltodict
from typing import Dict, List, Tuple
import json
import numpy as np

# I choose to work with the hebrew websites and not the English ons since
# IsraelHayom.com doesn't work good with RSS

DATA_SOURCE = {'https://www.haaretz.co.il/srv/htz---all-articles': 0,
               'https://www.israelhayom.co.il/rss.xml': 1}
CLASSIFICATIONS = ['Haaretz', 'IsraelHayom']
DATA_PATH = "./data"
UPDATE_INTERVAL = 1800
MODELS = [("DecisionTreeClassifier", lambda x: DecisionTreeClassifier(max_depth=x), range(1, 10)),
          ("RandomForestClassifier", lambda x: RandomForestClassifier(max_depth=x, n_estimators=10), range(1, 10)),
          ("SGDClassifier", lambda x: SGDClassifier(loss='hinge', penalty='l2',
                                                    alpha=x, random_state=42), [i * 1e-3 for i in range(2, 4)]),
          ("KNeighborsClassifier", lambda x: KNeighborsClassifier(x), range(3, 5)),
          ("SVC liner", lambda x: SVC(kernel="linear", C=x), [i * 0.125 for i in range(1, 8)]),
          ("AdaBoostClassifier", lambda x: AdaBoostClassifier(), [0]),
          ("MultinomialNB", lambda x: MultinomialNB(), [0]),
          ("SVC", lambda x: SVC(gamma=x, C=1), range(1, 8))]


def probe_website(url: str, classification: int) -> Dict[str, int]:
    """
    This function will probe in a website to search for titles
    :param url: the url of the website to probe in
    :param classification: The classification to set to the title found
    :return: Dictionary of the title to the classification
    """
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    # Get string data with utf 8 encoding
    data_str = str(response.data.decode('utf-8'))
    # Clean website errors of rss savings
    data_clean = data_str[
                 data_str.find('<rss'):data_str.find('</rss>') + 6].replace(
        '<iframe title=', '').replace('\\n', '\n')
    # Get relevant data from the string
    data = xmltodict.parse(data_clean)
    data = data['rss']['channel']['item']
    # Map the titles to the classification and return
    return {item['title']: classification for item in data}


def dump_to_file(data: Dict[str, int], path) -> None:
    """
    This function will dump the data to file as json
    :param data: data to dump
    :param path: path to dump to
    """
    with open(path, "w", encoding='utf8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_dump(path: str) -> Dict[str, int]:
    """
    This function will read from json dump
    :param path: the path of the json file
    :return: Dictionary of the titles to their classifications
    """
    try:
        with open(path, encoding='utf8') as file:
            return json.load(file)
    except:
        return dict()


def update_dataset(path: str, resources: Dict[str, int]) -> None:
    """
    This function will update the dataset with last titles from the web
    :param resources: The webs resources for the dataset and their
    classifications
    :param path: The path to the dataset file
    """
    data = read_dump(path)
    print(f"Before update: {len(data)}")
    for url, c in resources.items():
        probed = probe_website(url, c)
        data.update(probed)
    print(f"After update at {datetime.now()} got: {len(data)}")
    dump_to_file(data, path)


def read_data(path: str) -> np.array:
    """
    This function will read the dataset and return np array of the dataset
    :param path: path to json file of the dataset
    :return: np array of the dataset
    """
    data = read_dump(path)
    return np.array([list(data.keys()), list(data.values())])


def preprocessing(data: np.array, count_vect: CountVectorizer, tfidf_transformer: TfidfTransformer) -> Tuple[
    np.array, np.array,
    np.array, np.array]:
    """
    This function will preprocess the data from text to numbers in order to preform data learning
    :param data: array of text title and classifications
    :param count_vect: count vectorize object for mapping text to fetchers
    :param tfidf_transformer: make the mapping more accurate
    :return: The data splited to train and test
    """
    dataset = data.T
    # split to test and train
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, 0], dataset[:, 1],
                                                        test_size=0.1, shuffle=False)
    X_train = count_vect.fit_transform(X_train)
    X_test = count_vect.transform(X_test)
    X_train = tfidf_transformer.fit_transform(X_train)
    X_test = tfidf_transformer.transform(X_test)
    return X_test, X_train, y_test, y_train


def find_hyper(X_train: np.array, y_train: np.array, model_init, hypers: List[float]) -> float:
    """
    This function will choose hyperparameter for the model
    :param X_train: training set for the model
    :param y_train: training set for the model
    :param model_init: the model to find hyperparameter for
    :param hypers: the hyperparameters to choose from
    :return: the best hyperparameter for the model
    """
    scores = []
    for p in hypers:
        clf = model_init(p)
        s = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        scores.append(s.mean())
    return hypers[np.array(scores).argmax()]


def model_selection(X_test: np.array, X_train: np.array, y_test: np.array, y_train: np.array) -> BaseEstimator:
    """
    This function will select a model to predict the publisher of the news title
    :param X_test: The test set
    :param X_train: the train set
    :param y_test: the test labels
    :param y_train: the train labels
    :return: The chosen model fitted
    """
    best_score = 0
    best_model = None
    best_model_name = ""
    for name, model, hypers in MODELS:
        p = find_hyper(X_train, y_train, model, hypers)
        m = model(p)
        m.fit(X_train, y_train)
        train_predicted = m.predict(X_train)
        test_predicted = m.predict(X_test)
        test_score = np.mean(y_test == test_predicted)
        train_score = np.mean(y_train == train_predicted)
        if test_score > best_score:
            best_score, best_model, best_model_name = test_score, m, name
        print(f"{name}- Test score: {round(test_score * 100, 2)}%, Train score:"
              f" {round(train_score * 100, 2)}%.")
    print(f"\nWinning model is {best_model_name} with accuracy rate: {round(100 * best_score, 2)}%.")
    return best_model


def serialize_model(path: str, model: BaseEstimator, count_vect: CountVectorizer,
                    transformer: TfidfTransformer) -> None:
    """
    This function will serialize the model for future use
    :param path: path to save the model (must be directory)
    :param model: the main model
    :param count_vect: the count vectorize model
    :param transformer: the tfidf transformer model
    """
    with open(path + "/model", 'wb') as file:
        pickle.dump(model, file)
    with open(path + "/count", 'wb') as file:
        pickle.dump(count_vect, file)
    with open(path + "/transformer", 'wb') as file:
        pickle.dump(transformer, file)


def deserialize_model(path: str) -> Tuple[BaseEstimator, CountVectorizer, TfidfTransformer]:
    """
    This function will deserialize the model
    :param path: path where the serialized model is saved
    :return: fitted model, countvectorize and tfidftransformer to transform text to number fetchers
    """
    model, count_vect, transformer = None, None, None
    try:
        with open(path + "/model", 'rb') as file:
            model = pickle.load(file)
        with open(path + "/count", 'rb') as file:
            count_vect = pickle.load(file)
        with open(path + "/transformer", 'rb') as file:
            transformer = pickle.load(file)
    except:
        print(f"No model loaded, run the script with fit first in order to predict", file=sys.stderr)
    return model, count_vect, transformer


def fit_news_classifier(data_path: str) -> None:
    """
    This function will find, fit and save classifier for the news title problem
    :param data_path: path to where the dataset is saved, and where to save the result
    """
    data = read_data(data_path + "/dataset.json")
    count_vect, transformer = CountVectorizer(), TfidfTransformer()
    X_test_tfidf, X_train_tfidf, y_test, y_train = preprocessing(data, count_vect, transformer)
    model = model_selection(X_test_tfidf, X_train_tfidf, y_test, y_train)
    serialize_model(data_path, model, count_vect, transformer)


def predict(model_path: str, test_path: str) -> float:
    """
    This function will predict where the titles saved in the test_path file where originally published
    :param model_path: path where the model saved
    :param test_path: the titles to predict
    :return: success rate of the prediction
    """
    model, count_vect, transformer = deserialize_model(model_path)
    test_data = read_data(test_path).T
    X, y = test_data[:, 0], test_data[:, 1]
    test_count = count_vect.transform(X)
    test_trans = transformer.transform(test_count)
    result = model.predict(test_trans)
    for idx, prediction in enumerate(result):
        print(
            f"{CLASSIFICATIONS[int(prediction)] == CLASSIFICATIONS[int(y[idx])]}: The title was classified to be "
            f"{CLASSIFICATIONS[int(prediction)]} "
            f"newspaper.\n and it's from {CLASSIFICATIONS[int(y[idx])]}.")
    return float(np.mean(result == y))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'update':
            update_dataset(DATA_PATH + "/dataset.json", DATA_SOURCE)
        if sys.argv[1] == 'fit':
            fit_news_classifier(DATA_PATH)
        if sys.argv[1] == 'predict' and len(sys.argv) > 2:
            print(predict(DATA_PATH, sys.argv[2]))
        if sys.argv[1] == 'load_test':
            update_dataset(DATA_PATH + "/test.json", DATA_SOURCE)
    else:
        print(f"Arguments not in the current format. \nFormat: [update\\fit\\predict] [title (only for predict)]",
              file=sys.stderr)
