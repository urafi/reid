from os import listdir
import json
from tqdm import tqdm
import re

path = 'market/bounding_box_train'

train_images = listdir(path)

train_images = sorted(train_images)

ret = []
pattern = re.compile(r'([-\d]+)_c(\d)')
relabel = True
all_pids = {}

for im in tqdm(train_images):

    pid, cam = map(int, pattern.search(im).groups())
    if pid == -1: continue
    if relabel:
        if pid not in all_pids:
            all_pids[pid] = len(all_pids)
    else:
        if pid not in all_pids:
            all_pids[pid] = pid
    pid = all_pids[pid]
    cam -= 1
    ret.append((im, pid, cam))

with open("data/market_train.json", 'w') as write_file:
    json.dump(ret, write_file)
