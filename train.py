from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from torch.utils.data import DataLoader

import random
import torch.backends.cudnn as cudnn
import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.nn import DataParallel
import cv2
import numpy as np
from resnet import resnet50
from data_utils import train
import matplotlib.pyplot as plt


class Market(Dataset):

    def __init__(self, dtype):

        self.name = 'Coco'
        self.dtype = dtype
        self.crop_size = [256, 128]
        self.is_train = (dtype == 'train')

        if self.is_train:
            with open('data/market_train.json') as anno_file:
                self.anno = json.load(anno_file)
        else:

            with open('data/COCO2017_train.json') as anno_file:
                self.anno = json.load(anno_file)
            #self.img_dir = img_dir

    def __len__(self):

        length = len(self.anno)
        return length

    def apply_augmentation(self, example, is_train):

        im = cv2.imread(example['image'], 1)

        im_width, im_height = im.shape[1], im.shape[0]

        crop_pos = [im_width/2, im_height/2]
        max_d = np.maximum(im_width, im_height)
        scale = max(self.crop_size)/ max_d

        param = {'rot': 0,
                 'scale': scale,  # scale,
                 'flip': 0,
                 'tx': 0,
                 'ty': 0}

        if is_train:

            np.random.seed()
            param['scale'] *= (np.random.random() + .5)
            param['flip'] = np.random.binomial(1, 0.5)
            param['rot'] = (np.random.random() * (40 * 0.0174532)) - 20 * 0.0174532
            param['tx'] = np.int8((np.random.random() * 10) - 5)
            param['ty'] = np.int8((np.random.random() * 10) - 5)

        a = param['scale'] * np.cos(param['rot'])
        b = param['scale'] * np.sin(param['rot'])

        shift_to_upper_left = np.identity(3)
        shift_to_center = np.identity(3)

        t = np.identity(3)
        t[0][0] = a
        if param['flip']:
            t[0][0] = -a

        t[0][1] = -b
        t[1][0] = b
        t[1][1] = a

        shift_to_upper_left[0][2] = -crop_pos[0] + param['tx']
        shift_to_upper_left[1][2] = -crop_pos[1] + param['ty']
        shift_to_center[0][2] = self.crop_size[1]/2
        shift_to_center[1][2] = self.crop_size[0]/2
        t_form = np.matmul(t, shift_to_upper_left)
        t_form = np.matmul(shift_to_center, t_form)

        im_cv = cv2.warpAffine(im, t_form[0:2, :], (self.crop_size[1], self.crop_size[0]))
        img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float()
        img = torch.transpose(img, 1, 2)
        img = torch.transpose(img, 0, 1)
        img /= 255

        cls = int(example['class']) - 1

        return img, cls

    def __getitem__(self, ind):

            example = self.anno[ind]

            img, cls = self.apply_augmentation(example, self.is_train)

            return img, cls




def main():

    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    cudnn.benchmark = True
    #cudnn.deterministic = False
    cudnn.enabled = True

    marketTrain = Market('train')
    #coco_val = Coco('../Coco/val2017/', 'val',  transforms.Compose([normalize]))
    train_batch_size = 96
    #test_batch_size = 64
    train_loader = DataLoader(marketTrain, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=False)

    num_epochs = 130

    reidNet = resnet50(pretrained=True)
    reidNet = reidNet.cuda()
    #model = DataParallel(reidNet)
    model = reidNet

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean').cuda()

    start_epoch = 0

    for epoch in range(start_epoch, num_epochs):

        print( "Starting Epoch [%d]" % (epoch))

        #if epoch == 160:
        #    for param_group in optimiser.param_groups:
        #        param_group['lr'] = 1e-5
        #if epoch == 171:
        #    for param_group in optimiser.param_groups:
        #        param_group['lr'] = 1e-6

        tloss = train(train_loader, model, optimiser, criterion)
        with open('losses/loss_training.txt', 'a') as the_file:
            the_file.write(str(tloss) + '\n')
        the_file.close()
        #state = test(val_loader, model, epoch)
        #with open('losses/OKS2.txt', 'a') as the_file:
        #    the_file.write(str(state) + '\n')
        #the_file.close()
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimiser.state_dict(),
        }

        model_name = 'models_epoch/reidNet_' + str(epoch) + '.pth'
        torch.save(state, model_name)


if __name__ == '__main__':
    main()
