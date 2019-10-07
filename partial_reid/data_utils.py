import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import time
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from associate import get_preds

import json


def plot_keypoints(im, keypoints, scores):

    for j in range(17):
        if scores[j] > 0.1:
            x = int(keypoints[0][j])
            y = int(keypoints[1][j])
            im = cv2.circle(im, (int(x), int(y)), 2, (255, 0, 255), -1)

    cv2.imshow('Image_with_keypoints :', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_transform(param, crop_pos, output_size, scales):
    shift_to_upper_left = np.identity(3)
    shift_to_center = np.identity(3)

    a = scales[0] * param['scale'] * np.cos(param['rot'])
    b = scales[1] * param['scale'] * np.sin(param['rot'])

    t = np.identity(3)
    t[0][0] = a
    if param['flip']:
        t[0][0] = -a

    t[0][1] = -b
    t[1][0] = b
    t[1][1] = a

    shift_to_upper_left[0][2] = -crop_pos[0] + param['tx']
    shift_to_upper_left[1][2] = -crop_pos[1] + param['ty']
    shift_to_center[0][2] = output_size / 2
    shift_to_center[1][2] = output_size / 2
    t_form = np.matmul(t, shift_to_upper_left)
    t_form = np.matmul(shift_to_center, t_form)

    return t_form


def apply_augmentation(img, output_size, keypoints, scores):

    im = cv2.imread(img, 1)
    height, width = im.shape[:2]
    crop_pos = [int(width / 2), int(height / 2)]
    max_d = np.maximum(height, width)
    scales = [output_size / float(max_d), output_size / float(max_d)]

    param = {'rot': 0,
             'scale': 1,
             'flip': 0,
             'tx': 0,
             'ty': 0}

    t_form = get_transform(param, crop_pos, output_size, scales)
    im_cv = cv2.warpAffine(im, t_form[0:2, :], (output_size, output_size))
    img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    queries = torch.zeros(17, 2)

    X = keypoints[0:51:3]
    Y = keypoints[1:51:3]

    for j in range(17):
        queries[j, 0] = int(Y[j]/4)
        queries[j, 1] = int(X[j]/4)

    #plot_keypoints(img, [X, Y], scores)

    img = torch.from_numpy(img).float()
    img = torch.transpose(img, 1, 2)
    img = torch.transpose(img, 0, 1)
    img /= 255

    return img, queries


