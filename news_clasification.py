import sys
from datetime import datetime

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import SGDClassifier
import urllib3
import xmltodict
from typing import Dict
import json
import numpy as np

# I choose to work with the hebrew websites and not the English ons since
# IsraelHayom.com doesn't work good with RSS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_SOURCE = {'https://www.haaretz.co.il/srv/htz---all-articles': 0,
               # 'https://www.haaretz.com/srv/haaretz-latest-headlines': 0,
               # 'https://www.israelhayom.com/category/news/feed/':1,
               'https://www.israelhayom.co.il/rss.xml': 1}
DATA_PATH = "C:/Users/Ofer Ezrachi/Documents/Python Scripts/" \
            "News_title_classification/data/dataset.json"
UPDATE_INTERVAL = 1800


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
    with open(path, encoding='utf8') as file:
        return json.load(file)


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


def check_model_succses(X_train, y_train, baseModel, X_test, y_test):
    model = baseModel.fit(X_train, y_train)
    predicted = model.predict(X_test)
    return np.mean(predicted == y_test)


def choose_model():
    count_vect, tfidf_transformer = CountVectorizer(), TfidfTransformer()
    dataset = read_data(DATA_PATH).T
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, 0],
                                                        dataset[:, 1],
                                                        test_size=0.1,
                                                        shuffle=False)
    X_train_counts = count_vect.fit_transform(X_train)
    X_test_counts = count_vect.transform(X_test)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    print(check_model_succses(X_train_tfidf, y_train,
                              SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42),
                              # MultinomialNB(),
                              # DecisionTreeClassifier(max_depth=5),
                              # KNeighborsClassifier(1),
                              # RandomForestClassifier(max_depth=5,
                              #                        n_estimators=10,
                              #                        max_features=1),
                              # SVC(kernel="linear", C=0.025),
                              # SVC(gamma=2, C=1),
                              X_test_tfidf, y_test))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'update':
        update_dataset(DATA_PATH, DATA_SOURCE)
    else:
        choose_model()
