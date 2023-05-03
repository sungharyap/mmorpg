from utils import get_news
import json

TEXT_FILE = "20210101_20210228.txt"
# TEXT_FILE = "tmp_data.txt"
JSON_FILE = "data_small.json"

with open(TEXT_FILE, "r") as f:
    url_list = f.read().split('\n')

news_list = get_news(url_list)
news_dict = {"data" : []}
news_dict['data'] = [news.get_json() for news in news_list]

with open(JSON_FILE, "w") as f:
    json.dump(news_dict, f, ensure_ascii=False, indent=4)
