from os import listdir
import json
from tqdm import tqdm

path = 'market/bounding_box_train'

train_images = listdir(path)

train_images = sorted(train_images)

rec = []

for im in tqdm(train_images):

    rec.append({'image': path + '/' + im,
                'class': int(im.split('_')[0])
                })

with open("data/market_train.json", 'w') as write_file:
    json.dump(rec, write_file)
