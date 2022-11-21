from datetime import datetime
from time import sleep

import urllib3
import xmltodict
from typing import Dict
import json

DATA_SOURCE = ['https://www.haaretz.com/srv/haaretz-latest-headlines',
               'https://www.israelhayom.co.il/rss.xml']
DATA_PATH = "D:/הפקות ופרויקטים/Pythone projects/News_title_classification/data/dataset.json"


def probe_website(url: str, clasification: int) -> Dict[str, int]:
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    data_str = str(response.data.decode('utf-8'))
    data_clean = data_str[
                 data_str.find('<rss'):data_str.find('</rss>') + 6].replace(
        '<iframe title=', '').replace('\\n', '\n')
    data = xmltodict.parse(data_clean)
    data = data['rss']['channel']['item']

    return {item['title']: clasification for item in data}


def dump_to_file(data: Dict[str, int], path):
    with open(path, "w", encoding='utf8') as file:
        json.dump(data, file, ensure_ascii=False)


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
    with open('./file.txt') as file:
        file.write("Hello")