import os
import json
import pandas as pd
import tqdm

with open(os.path.join('data_small.json')) as f:
    obj = json.load(f)['data']


img_name = []
text = []

for e in tqdm.tqdm(obj):
    text.append(e['text'].split("[SEP]")[0])
    img_name.append(e['img'])

df = pd.DataFrame()
df['file_name'] = img_name
df['text'] = text

df.to_csv('metadata.csv', index=False)