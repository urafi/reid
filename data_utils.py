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
from torch.autograd import  Variable
import json

def get_preds(prs, sr, mat):

    pool = torch.nn.MaxPool2d(3, 1, 1).cuda()

    xoff = sr[0:17]
    yoff = sr[17:34]

    #prs2 = prs

    o = pool(Variable(prs.cuda())).data.cpu()
    maxm = torch.eq(o, prs).float()
    prs = prs * maxm

    prso = prs.view(17, 64 * 64)
    val_k, ind = prso.topk(1, dim=1)
    xs = ind % 64
    ys = (ind / 64).long()


    keypoints = []
    score = 0
    points = torch.zeros(17, 2)
    c = 0
    v_score = np.zeros(17)

    for j in range(17):

        x, y = xs[j][0], ys[j][0]
        dx = xoff[j][int(y)][int(x)]
        dy = yoff[j][int(y)][int(x)]
        points[j][0] = (x * 4) + dx.item()
        points[j][1] = (y * 4) + dy.item()
        v_score[j] = val_k[j][0]
        c += 1

    score /= c

    for j in range(17):

        point = torch.ones(3, 1)
        point[0][0] = points[j][0]
        point[1][0] = points[j][1]

        keypoint = np.matmul(mat, point)
        keypoints.append(float(keypoint[0].item()))
        keypoints.append(float(keypoint[1].item()))
        keypoints.append(v_score[j])

    return keypoints, score

def plot_masks_on_image(im, masks):
    n_masks = masks.shape[0]
    for j in range(n_masks):
        r = cv2.resize(masks[j].numpy(), (128, 256))
        im[0, :, :] = im[0, :, :] + torch.from_numpy(r)

    return im

def learning_rate_list(lr=.01, decay_steps=10, decay_rate=0.95, total_epochs=60):

    decayed_lr = np.zeros((total_epochs,1))
    decayed_lr[0] = lr
    for ep in range(1, total_epochs):
        if ep % 5 == 0:
            tmp = int((ep) / decay_steps)
        else:
            tmp = 0
        #print(tmp)
        decayed_lr[ep][0] = decayed_lr[ep-1][0] * (pow(decay_rate, tmp))
    return decayed_lr


def preprocess_train(path, relabel):

    #process market dataset
    train_images = listdir(path)

    train_images = sorted(train_images)
    ret = []
    pattern = re.compile(r'([-\d]+)_c(\d)')
    relabel = relabel
    all_pids = {}

    print('Preprocessing Market data')

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
        ret.append(('market/bounding_box_train/' + im, pid, cam))

    print('Preprocessing PoseTrack')
    with open('data/PoseTrack_crop.json') as anno_file:
        annos = json.load(anno_file)

    total_pids = 1501
    N = 10000

    for i in range(N):

        pid = int(total_pids + annos[i]['class'])
        if pid not in all_pids:
            all_pids[pid] = len(all_pids)

        pid = all_pids[pid]
        cam = -1
        if annos[i]['image'] == 'poseTrack_crops/26325.jpg':
            continue
        ret.append((annos[i]['image'], pid, cam))

    return ret, int(len(all_pids))



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

        outputs = model(inputs)

        l = torch.sum(
            torch.stack([criterion(output, ids) for output in outputs]))


        total_loss += l.data.item()

        optimiser.zero_grad()
        l.backward()
        optimiser.step()

        #print ('TrainEpoch [%d], Iter [%d/%d] , loss [%f]'
        #       % (epoch + 1, i + 1, NBatches, l.data.item()))

    #print('Total Time for train epoch %.4f' % (time.time() - start_time))

    return total_loss


