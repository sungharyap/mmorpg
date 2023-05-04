import json
import os

JSON_FILE = 'data_small.json'

with open(JSON_FILE, 'r') as f:
    data = json.load(f)
    
for news in data['data']:
    # print(news['img'])
    if not os.path.exists(f"img/{news['img']}"):
        os.system(f"curl -s {news['img_url']} > img/{news['img']}")