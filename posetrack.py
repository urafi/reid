from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random
import argparse
import os
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import  tqdm

with open('data/PoseTrack_train.json') as anno_file:
    annos = json.load(anno_file)

N = len(annos)

rec = []

ids  = {}

for i in tqdm(range(N)):

    anno = annos[i]

    im = cv2.imread('../PoseTrack/' + anno['image'], 1)

    x1, x2 = int(anno['bbox'][0]), int(anno['bbox'][0] + anno['bbox'][2])
    y1, y2 = int(anno['bbox'][1]), int(anno['bbox'][1] + anno['bbox'][3])

    if (x2-x1) == 0 or (y2-y1)==0:
        continue

    crop = im[y1:y2,x1:x2,:]

    path = 'poseTrack_crops/' + str(i) + '.jpg'

    rec.append({'image' : path, 'class': anno['label']})

    cv2.imwrite(path, crop)


with open("data/PoseTrack_crop.json" , 'w') as write_file:
   json.dump(rec, write_file)
