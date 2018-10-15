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
from data_utils import *
import matplotlib.pyplot as plt
from reid.evaluators import *
import argparse


class Market(Dataset):

    def __init__(self, dtype, data, img_dir, name):

        self.name = name
        self.dtype = dtype
        self.crop_size = [256, 128]
        self.is_train = (dtype == 'train')
        self.anno = data
        self.img_dir = img_dir

        #if self.is_train:
        #    with open('data/market_train.json') as anno_file:
        #        self.anno = json.load(anno_file)
        #else:

        #    with open('data/COCO2017_train.json') as anno_file:
        #        self.anno = json.load(anno_file)
            #self.img_dir = img_dir

    def __len__(self):

        length = len(self.anno)
        return length

    def apply_augmentation(self, example, is_train):

        im = cv2.imread(self.img_dir + example[0], 1)

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
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        img = normalize(img)

        #cls = example[1]

        return img, example[0], example[1], example[2]

    def __getitem__(self, ind):

            example = self.anno[ind]

            img, fname, pid, cam = self.apply_augmentation(example, self.is_train)

            return img, fname, pid, cam




def main(args):

    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    cudnn.benchmark = True
    #cudnn.deterministic = False
    cudnn.enabled = True

    train_source, num_classes = preprocess('market/bounding_box_train', relabel=True)
    gallery, _ = preprocess('market/bounding_box_test', relabel=False)
    query, _ = preprocess('market/query', relabel=False)

    marketTrain = Market('train', train_source, 'market/bounding_box_train/','train')
    galleryds = Market('val', gallery, 'market/bounding_box_test/','gallery')
    querds = Market('val', query, 'market/query/', 'query')

    #coco_val = Coco('../Coco/val2017/', 'val',  transforms.Compose([normalize]))
    train_batch_size = 32
    #test_batch_size = 64
    train_loader = DataLoader(marketTrain, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=False)
    query_loader = DataLoader(querds, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=False)
    gallery_loader = DataLoader(galleryds, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=False)


    num_epochs = 60

    reidNet = resnet50(pretrained=True, num_classes=num_classes)
    reidNet = reidNet.cuda()
    #model = DataParallel(reidNet)
    model = reidNet

    lr = 1e-2

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, eps=.1)

    #checkpoint = torch.load('models_epoch/reidNet_10.pth')
    #model.load_state_dict(checkpoint['state_dict'])
    #optimiser.load_state_dict(checkpoint['optimizer'])

    criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean').cuda()

    start_epoch = 0#checkpoint['epoch'] + 1

    for epoch in range(start_epoch, num_epochs):

        if epoch > 0 and (epoch % 10 == 0):
                lr /= 2
                for param_group in optimiser.param_groups:
                    param_group['lr'] = lr


        print( "Starting Epoch [%d]" % (epoch))


        tloss = train(train_loader, model, optimiser, criterion)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimiser.state_dict(),
        }

        evaluator = Evaluator(model)
        all = evaluator.evaluate(query_loader, gallery_loader, query, gallery, args.output_feature, args.rerank)
        with open('losses/rank1.txt', 'a') as the_file:
            the_file.write(str(all[0] * 100) + '\n')
        the_file.close()

        model_name = 'models_epoch/reidNet_' + str(epoch) + '.pth'
        torch.save(state, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="baseline")
    # source
    parser.add_argument('-s', '--source', type=str, default='market',
                        choices=['market', 'duke', 'cuhk03_detected'])
    # target
    parser.add_argument('-t', '--target', type=str, default='market',
                        choices=['market', 'duke'])
    # images
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size for source")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    #parser.add_argument('-a', '--arch', type=str, default='resnet50',
    #                    choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--print-freq', type=int, default=100)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    #working_dir = osp.dirname(osp.abspath(__file__))
    # parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='//media/ekodirov/1002d198-cc12-4f27-aae2-fdef0f8cea56/Anvpersons/ReID_datasets/deep-person-reid-datasets/hhl/data/')
    #parser.add_argument('--logs-dir', type=str, metavar='PATH',
    #                    default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    #  perform re-ranking
    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")

    main(parser.parse_args())
