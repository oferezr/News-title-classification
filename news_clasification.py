from datetime import datetime
from time import sleep

import urllib3
import xmltodict
from typing import Dict
import json

DATA_SOURCE = ['https://www.haaretz.com/srv/haaretz-latest-headlines',
               'https://www.israelhayom.co.il/rss.xml']
DATA_PATH = "C:/Users/Ofer Ezrachi/Documents/Python Scripts/News_title_classification/data/dataset.json"


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


def dump_to_file(data: Dict[str, int], path)->None:
    """
    This function will dump the data to file as json
    :param data: data to dump
    :param path: path to dump to
    """
    with open(path, "w", encoding='utf8') as file:
        json.dump(data, file, ensure_ascii=False,indent=2)


def read_dump(path: str) -> Dict[str, int]:
    with open(path, encoding='utf8') as file:
        return json.load(file)


def update_dataset(path: str):
    data = read_dump(path)
    print(f"Before update: {len(data)}")
    for i in range(len(DATA_SOURCE)):
        probed = probe_website(DATA_SOURCE[i], i)
        data.update(probed)
    print(f"After update at {datetime.now()} got: {len(data)}")
    dump_to_file(data, path)


if __name__ == '__main__':
    update_dataset(DATA_PATH)
