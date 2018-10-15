import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import time
import torchvision
from tqdm import tqdm
from os import listdir
import re


def learning_rate_list(lr=1e-2, decay_steps=9, decay_rate=0.95, total_epochs=60):

    decayed_lr = np.zeros((total_epochs,1))
    decayed_lr[0] = lr
    for ep in range(1, total_epochs):
        tmp = int((ep) / decay_steps)
        #print(tmp)
        decayed_lr[ep][0] = decayed_lr[ep-1] * (pow(decay_rate, tmp))
    return decayed_lr

def preprocess(path, relabel):

    train_images = listdir(path)

    train_images = sorted(train_images)
    ret = []
    pattern = re.compile(r'([-\d]+)_c(\d)')
    relabel = relabel
    all_pids = {}

    print('Preprocessing Training data')

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

    return ret, int(len(all_pids))

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


def train(train_loader, model, optimiser, criterion):

    # switch to train mode
    model.train()
    total_loss = 0
    start_time = time.time()

    for i, (img, fname, pid, cam) in enumerate(tqdm(train_loader)):

        ids = pid.cuda(non_blocking=True)
        inputs = img.cuda(non_blocking=True)

        output = model(inputs)

        l = criterion(output, ids)

        total_loss += l.data.item()

        optimiser.zero_grad()
        l.backward()
        optimiser.step()

        #print ('TrainEpoch [%d], Iter [%d/%d] , loss [%f]'
        #       % (epoch + 1, i + 1, NBatches, l.data.item()))

    print('Total Time for train epoch %.4f' % (time.time() - start_time))

    return total_loss


