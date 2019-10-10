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
from bn_inception2 import bninception
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

flipRef = [i-1 for i in [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16] ]

class ReidData:

    def __init__(self, img_dir=None):

        self.image_names = os.listdir(img_dir)
        #with open('data/COCO2017_val.json') as anno_file:
        #    self.anno = json.load(anno_file)
        self.img_dir = img_dir

    def __len__(self):

        return len(self.image_names)

    def apply_augmentation_test(self, img, output_size):

        im = cv2.imread(img, 1)
        height, width = im.shape[:2]
        crop_pos = [int(width/2), int(height/2)]
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
        imf = cv2.flip(img, 1)

        img = torch.from_numpy(img).float()
        img = torch.transpose(img, 1, 2)
        img = torch.transpose(img, 0, 1)
        img /= 255

        imf = torch.from_numpy(imf).float()
        imf = torch.transpose(imf, 1, 2)
        imf = torch.transpose(imf, 0, 1)
        imf /= 255

        warp = torch.from_numpy(np.linalg.inv(t_form))

        return img, imf, warp

    def __getitem__(self, ind):

        #images, imagesf, warps = self.apply_augmentation_test(self.img_dir + self.image_names[ind], output_size=256)

        #kpts = torch.from_numpy(np.array(self.anno[ind]['keypoints'])).float()
        #area = torch.from_numpy(np.array(self.anno[ind]['area']))
        #bbx = torch.from_numpy(np.array(self.anno[ind]['bbox']))

        #meta = {'index': ind, 'imgID': self.anno[ind]['im_id'],
        #        'warps': warps, 'kpts': kpts, 'area' : area, 'bbox' : bbx, 'truncated':self.anno[ind]['truncated']}
        im, imf, warp = self.apply_augmentation_test(self.img_dir + self.image_names[ind], output_size=256)

        return im, imf, warp, self.image_names[ind]
        #return {'images': images, 'imagesf' : imagesf, 'meta' : meta}


manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
cudnn.benchmark = True


reid_data = ReidData(img_dir='PartialREID/whole_body_images/')
test_batch_size = 1
val_loader = DataLoader(reid_data, batch_size=test_batch_size, shuffle=False, num_workers=8)


poseNet = bninception()
poseNet = poseNet.cuda()
model = DataParallel(poseNet)
checkpoint = torch.load('models/model_r222.pth')
pretrained_dict = checkpoint['state_dict']
model.load_state_dict(pretrained_dict)
model.eval()


det = []
image_ids = []
im_counter = 1
total_loss = 0
counter = 0
total = 0
truncated = 0

for i_batch, (im, imf, warps, im_name) in enumerate(val_loader):

    with torch.no_grad():

        inputs = im.cuda(non_blocking=True)
        inputsf = imf.cuda(non_blocking=True)

        output = model(inputs)
        output_det = torch.sigmoid(output[1][:, 0:17, :, :])
        output_det = output_det.data.cpu()

        outputf = model(inputsf)
        output_detf = torch.sigmoid(outputf[1][:, 0:17, :, :])
        output_detf = output_detf.data.cpu()

        sr = output[1][:, 17:51, :, :].data.cpu()

        print('Iter [%d/%d]' %(i_batch + 1, len(reid_data) // test_batch_size))

        N = output_det.shape[0]

        for n in range(N):

            single_result_dict = {}

            prs = torch.zeros(17, 64, 64)
            output_detf[n] = output_detf[n][flipRef]
            for j in range(17):
                prs[j] = output_det[n][j] + torch.from_numpy(cv2.flip(output_detf[n][j].numpy(), 1))

            keypoints, score = get_preds(prs, sr[n], warps[n])

            single_result_dict['image'] = im_name[0]
            single_result_dict['keypoints'] = keypoints
            single_result_dict['score'] = score

            det.append(single_result_dict)

with open('whole_body_keypoints.json', 'w') as f:
        json.dump(det, f)


