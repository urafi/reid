from torch.utils.data import DataLoader
from data_utils import *
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
import torchvision
import random
import torch.backends.cudnn as cudnn
import pickle
import cv2
import json
from torch.nn import DataParallel
from associate import *
from embedding_siamese import Siamese
from embeddings_loss import  get_corr_maps3x3_fast
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'



manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
cudnn.benchmark = True

poseNet = Siamese()
model = DataParallel(poseNet)
model.cuda()
checkpoint = torch.load('models/m_coco_5x5_51.pth')
pretrained_dict = checkpoint['state_dict']
model.load_state_dict(pretrained_dict)
model.eval()

with open('data/partial_body_keypoints.json') as anno_file:
    probes = json.load(anno_file)

with open('data/whole_body_keypoints.json') as anno_file:
    galleries = json.load(anno_file)

queries_all = []
images_warped_probes = []
images_warped_gallery = []
match_scores = []
features_probe = []
features_gallery = []

print('Preprocessing Probe')

for p in tqdm(probes):

    im = 'PartialREID/partial_body_images/' + p['image']
    im, queries = apply_augmentation(im, 256, p['keypoints'], p['score'])
    images_warped_probes.append(im)
    queries_all.append(queries)
    match_scores.append({'keypoint_scores': p['score'], 'similarity_scores':[]})

print('\nPreprocessing Gallery')

for p in tqdm(probes):

    im = 'PartialREID/whole_body_images/' + p['image']
    im, _ = apply_augmentation(im, 256, p['keypoints'], p['score'])
    images_warped_gallery.append(im)

print('\nComputing features Probe')

with torch.no_grad():
    for im in tqdm(images_warped_probes):
        im = im.view(1, 3, 256, 256).cuda()
        f, _ = model(im, im)
        features_probe.append(f)

print('\nComputing features gallery')
with torch.no_grad():
    for im in tqdm(images_warped_gallery):
        im = im.view(1, 3, 256, 256).cuda()
        f, _ = model(im, im)
        features_gallery.append(f)

print('\n Computing matching scores')

with torch.no_grad():
    for i, fp in enumerate(tqdm(features_probe)):

        q = queries_all[i].view(1, 17, 2)
        for fg in features_gallery:
           correlations = get_corr_maps3x3_fast(fp, fg, q, match_scores[i]['keypoint_scores'])
           correlations2 = correlations.view(1, 17, 64 * 64).data.cpu()
           val_k, _ = correlations2.topk(1, dim=2)
           keypoint_matching_scores = val_k.data.cpu().view(17).numpy()
           #match_scores[i]['similarity_scores'].append(keypoint_matching_scores.tolist())
           match_scores[i]['similarity_scores'].append(float(keypoint_matching_scores.sum()))
       #print('done')

with open('match_scores.json', 'w') as f:
    json.dump(match_scores, f)




