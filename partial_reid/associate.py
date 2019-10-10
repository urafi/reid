import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pycocotools.mask as mask
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from data_utils import *

part_ref = {'ankle':[15,16],'knee':[13,14],'hip':[11,12],
            'wrist':[9,10],'elbow':[7,8],'shoulder':[5,6],
            'face':[0,1,2],'ears':[3,4]}
pairs = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11], [6, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
parents = [0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]

hieracrchy = {'1':0,'3':0,'5':0,'7':5,'9':7,'11':5,'13':11,'15':13,
              '2':0,'4':0,'6':0,'8':6,'10':8,'12':6,'14':12,'16':14}

flipRef = [i-1 for i in [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16] ]

graph = {'0': set(['1', '2', '3', '4', '5', '6', '11', '12']),
         '1': set(['0']),
         '2': set(['0']),
         '3': set(['0']),
         '4': set(['0']),
         '5': set(['0', '7']),
         '6': set(['0', '8']),
         '7': set(['5', '9']),
         '8': set(['6', '10']),
         '9': set(['7']),
         '10': set(['8']),
         '11': set(['0', '13']),
         '12': set(['0', '14']),
         '13': set(['11', '15']),
         '14': set(['12', '16']),
         '15': set(['13']),
         '16': set(['14']),
         }


def compute_OKS(gt, dt, bb, area):

    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    g = np.array(gt)
    xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
    k1 = np.count_nonzero(vg > 0)
    x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
    d = np.array(dt)
    xd = d[0::3]; yd = d[1::3]
    if k1>0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg
    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
        dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
    e = (dx**2 + dy**2) / vars / (area.item()+np.spacing(1)) / 2
    if k1 > 0:
        e=e[vg > 0]
    OKS = np.sum(np.exp(-e)) / e.shape[0]

    return OKS


def already_exists(idx, pos, instances):

    found = False

    xp = pos[0]
    yp = pos[1]

    found_id = -1
    for i,ins in enumerate(instances):

        x = ins['points'][idx][0]
        y = ins['points'][idx][1]
        dx = np.absolute(x-xp)
        dy = np.absolute(y-yp)
        if dx <= 9 and dy <= 9:
           found = True
           found_id = i

    return found, found_id


def get_max(val_k):

    val_k = val_k.numpy()
    ind = np.unravel_index(np.argmax(val_k, axis=None), val_k.shape)
    return ind


def get_preds(prs, sr, mat):


    xoff = sr[0:17]
    yoff = sr[17:34]


    res = 64
    prso = prs.view(17, res * res)
    val_k, ind = prso.topk(1, dim=1)
    xs = ind % res
    ys = (ind / res).long()

    keypoints = []
    scores = []
    points = torch.zeros(17, 2)

    for j in range(17):

        x, y = xs[j][0], ys[j][0]
        dx = xoff[j][int(y)][int(x)]
        dy = yoff[j][int(y)][int(x)]
        points[j][0] = (x * 4) + dx.item()
        points[j][1] = (y * 4) + dy.item()
        scores.append(val_k[j][0].item())

    X = []
    Y = []

    for j in range(17):

        point = torch.ones(3, 1)
        point[0][0] = points[j][0]
        point[1][0] = points[j][1]

        keypoint = np.matmul(mat, point)
        X.append([float(keypoint[0].item())])
        Y.append([float(keypoint[1].item())])

        #keypoints.append(float(keypoint[0].item()))
        #keypoints.append(float(keypoint[1].item()))
    keypoints.append([X, Y])

    return keypoints, scores


def get_preds2(prs, mat, sr, votes):

    pool = nn.MaxPool2d(3, 1, 1).cuda()

    xoff = sr[0:17]
    yoff = sr[17:34]

    ux1 = votes[0:16]
    uy1 = votes[16:32]

    ux2 = votes[32:48]
    uy2 = votes[48:64]

    prs2 = prs

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

    for j in range(17):

        if j > 0:

            parent = parents[j - 1]
            px, py = int(points[parent, 0]), int(points[parent, 1])
            prx = px + ux1[j-1][int(py)][int(px)].item()
            pry = py + uy1[j-1][int(py)][int(px)].item()

            prx = max(0, min(63, prx))
            pry = max(0, min(63, pry))

            yrs = range(max(0, int(pry) - 20), min(63, int(pry) + 20))
            xrs = range(max(0, int(prx) - 20), min(63, int(prx) + 20))
            conf = prs2[j][int(pry)][int(prx)]
            for _, yr in enumerate(yrs):
                for _, xr in enumerate(xrs):
                    if prs2[j][int(yr)][int(xr)] > conf:
                        prx = xr
                        pry = yr
                        conf = prs2[j][int(yr)][int(xr)]

            points[j][0] = prx
            points[j][1] = pry
            score += conf

        else:

            points[j][0] = xs[j]
            points[j][1] = ys[j]
            score += val_k[j]


    for j in range(17):

        x, y = points[j][0], points[j][1]
        dx = xoff[j][int(y)][int(x)]
        dy = yoff[j][int(y)][int(x)]
        points[j][0] = (x * 4) + dx.item()
        points[j][1] = (y * 4) + dy.item()
        #score += val_k[j][0]
        c += 1

    score /= c

    for j in range(17):

        point = torch.ones(3, 1)
        #if points[j][0] > 0 and points[j][1] > 0:
        point[0][0] = points[j][0]
        point[1][0] = points[j][1]
        #else:
        #    point[0][0] = xm
        #    point[1][0] = ym

        keypoint = np.matmul(mat, point)
        keypoints.append(float(keypoint[0].item()))
        keypoints.append(float(keypoint[1].item()))
        #keypoints.append(int(point[0][0]))
        #keypoints.append(int(point[1][0]))
        keypoints.append(1)

    return keypoints, score.item()




def get_pos(connections, vertex, votes, pos, prs):

    nodes_pos = []
    if pos[0] > 0 and pos[1] > 0 and pos[0] < 64 and pos[1] < 64:

        y, x = int(np.round(pos[1])), int(np.round(pos[0]))
        vertex = int(vertex)

        for _, c in enumerate(connections):

            c = int(c)

            yrs = range(max(0, y-10), min(63, y+10))
            xrs = range(max(0, x-10), min(63, x+10))
            nx = np.zeros([len(yrs)*len(xrs)])
            ny = np.zeros([len(yrs)*len(xrs)])
            counter = 0
            conf = 0
            mx = 0
            my = 0

            id = np.maximum(vertex, c)-1
            if vertex > c:
                for _, yr in enumerate(yrs):
                    for _, xr in enumerate(xrs):
                        dx = votes[1][0][id][yr][xr].item()
                        dy = votes[1][1][id][yr][xr].item()
                        #nx[counter] = dx + xr
                        #ny[counter] = dy + yr
                        if prs[vertex][yr][xr] > conf :
                            conf = prs[vertex][yr][xr]
                            mx = dx + xr
                            my = dy + yr
                        counter += 1

            else:

                for _, yr in enumerate(yrs):
                    for _, xr in enumerate(xrs):

                        dx = votes[0][0][id][yr][xr]
                        dy = votes[0][1][id][yr][xr]
                        #nx[counter] = dx + xr
                        #ny[counter] = dy + yr
                        if prs[vertex][yr][xr] > conf :
                            conf = prs[vertex][yr][xr]
                            mx = dx + xr
                            my = dy + yr

                        counter += 1

            #nx /= counter
            #ny /= counter
            nx = mx#np.round(np.median(nx))
            ny = my#np.round(np.median(ny))

            if nx < 0 or nx > 63 or ny < 0 or ny > 63:
                nodes_pos.append([-1, -1])
                continue

            nx = np.maximum(0,np.minimum(nx, 63))
            ny = np.maximum(0,np.minimum(ny, 63))
            nodes_pos.append([nx,ny])

    return nodes_pos


def get_other_joints(start, votes, pos, sroff, prs):

    visited, queue = set(), [str(start)]
    points = torch.ones(17,2).mul(-1)
    dx = sroff[start][pos[1]][pos[0]]
    dy = sroff[start+17][pos[1]][pos[0]]
    pos[0] = round((pos[0].item()*4 + dx.item())/4)
    pos[1] = round((pos[1].item() * 4 + dy.item()) / 4)
    points[start][0] = pos[0]
    points[start][1] = pos[1]
    visiting_order = []

    while queue:
        pos = []
        vertex = queue.pop(0)
        pos.append(points[int(vertex)][0])
        pos.append(points[int(vertex)][1])
        if pos[0] == -1 or pos[1] == -1:
            continue
        #visiting_order.append(vertex)
        if vertex not in visited:
            visited.add(vertex)
            connections = graph[vertex] - visited
            nodes_pos = get_pos(connections, vertex, votes, pos, prs)
            queue.extend(connections)
            if len(nodes_pos) > 0:
                for i, c in enumerate(connections):
                    c = int(c)
                    y, x = nodes_pos[i][1], nodes_pos[i][0]
                    if y == -1 or x == -1:
                        continue
                    yrs = range(max(0, int(y) - 10), min(63, int(y) + 10))
                    xrs = range(max(0, int(x) - 10), min(63, int(x) + 10))
                    nx = np.zeros([len(yrs) * len(xrs)])
                    ny = np.zeros([len(yrs) * len(xrs)])
                    counter = 0
                    mx = 0
                    my = 0
                    conf = 0
                    for _, yr in enumerate(yrs):
                        for _, xr in enumerate(xrs):
                            if prs[c][int(yr)][int(xr)] > conf:
                                conf = prs[c][int(yr)][int(xr)]
                                ox = sroff[c][int(yr)][int(xr)]
                                oy = sroff[c + 17][int(yr)][int(xr)]
                                mx = xr * 4 + ox
                                my = yr * 4 + oy
                            #nx[counter] = xr * 4 + ox
                            #ny[counter] = yr * 4 + oy
                            #counter += 1
                    #dx = sroff[c][y][x]
                    #dy = sroff[c + 17][y][x]
                    rx = np.round(np.median(mx)/4)#round((nodes_pos[i][0]*4 + dx)/4)
                    ry = np.round(np.median(my)/4)#round((nodes_pos[i][1]*4 + dy) / 4)
                    points[c][0] = rx
                    points[c][1] = ry

    #print(visiting_order)
    return points

def get_torso(p1,p2):


    X = np.abs(p1[0] - p2[0]) * np.abs(p1[0] - p2[0])
    Y = np.abs(p1[1] - p2[1]) * np.abs(p1[1] - p2[1])
    torso = np.sqrt(X + Y)

    return torso


def get_range(I):

    min = 999
    max = 0
    N = I.shape[0]
    r = [0, 0]

    for n in range(N):

        if I[n] <= 0:
            continue
        if I[n] > max:
            r[1] = n
            max = I[n]
        if I[n] < min:
            r[0] = n
            min = I[n]

    return r


def apply_augmentation_torso(example, output_size=256):

    im = cv2.imread(example['image'], 1)
    iw, ih = im.shape[1], im.shape[0]

    x1, x2 = int(example['bbox'][0]), int(example['bbox'][0]+example['bbox'][2])
    y1, y2 = int(example['bbox'][1]), int(example['bbox'][1] + example['bbox'][3])

    imcrop = im[y1:y2,x1:x2,:]

    crop_pos =  [example['bbox'][2]/2, example['bbox'][3]/2]#example['center']

    X = example['keypoints'][0:51:3]  # ins[0,:]
    Y = example['keypoints'][1:51:3]  # ins[1,:]

    state = example['keypoints'][2:51:3]
    max_d = example['scale']
    scales = [output_size / float(max_d), output_size / float(max_d)]

    torso_visible = True

    if (state[5] == 0 or state[12] == 0) and (state[6] == 0 or state[11] == 0):
        torso_visible = False

    mT = 0

    joints = np.ones([3, 17])
    joints[0, :] = np.array(X) - x1
    joints[1, :] = np.array(Y) - y1


    if torso_visible:
        if state[5] > 0 and state[12] > 0 :
            tH = np.abs(joints[1][5] - joints[1][12])
            tW = np.abs(joints[0][5] - joints[0][12])
            mT = np.maximum(tH, tW)
        if state[6] > 0 and state[11] > 0:
            tH = np.abs(joints[1][6] - joints[1][11])
            tW = np.abs(joints[0][6] - joints[0][11])
            mT2 = np.maximum(tH, tW)
            if mT2 > mT:
                mT = mT2


    param = {'rot': 0,
             'scale': 1,
             'flip': 0,
             'tx': 0,
             'ty': 0}

    if torso_visible == False:
        print('pb')

    if torso_visible :

        mTscaled = scales[0] * mT


        rescale = 50/mTscaled

        scales[0] *= rescale
        scales[1] *= rescale


    t_form = get_transform(param, crop_pos, output_size, scales)
    joints_warped = np.matmul(t_form, joints)

    for j in range(17):
        if state[j] > 0 and (joints_warped[0][j] < 0 or joints_warped[1][j] < 0 or joints_warped[1][j] >= output_size or joints_warped[0][j] >= output_size):
            print('prob')

    parts = torch.zeros(2, 17)
    counters = torch.zeros(17)

    # for p in range(17):
    #
    #     if state[p] == 0:
    #         continue
    #
    #     parts[0][p] = joints_warped[0][p]
    #     parts[1][p] = joints_warped[1][p]
    #     counters[p] = 1

    im_cv = cv2.warpAffine(imcrop, t_form[0:2, :], (output_size, output_size))
    img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(img).float()
    #img = torch.transpose(img, 1, 2)
    #img = torch.transpose(img, 0, 1)
    #img /= 255

    #imf = torch.from_numpy(imf).float()
    #imf = torch.transpose(imf, 1, 2)
    #imf = torch.transpose(imf, 0, 1)
    #imf /= 255

    #warp = torch.from_numpy(np.linalg.inv(t_form))

    return img, torso_visible, max_d
