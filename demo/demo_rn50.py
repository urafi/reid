'''
    Training code
    @author: Elyor Kodirov
    @date: 19/02/2019
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel

from rn import ResNet50_pytorch


def read_img(path):
    img = plt.imread(path) / 255.0
    img = cv2.resize(img, (128, 384))
    return img

def main():
    load_weight_path = './ckpt/reidnet_smtr62_wfc_31.80379746835443.pth'
    model = ResNet50_pytorch()
    model = torch.nn.DataParallel(model).cuda()

    # use pretrained model

    model_dict = model.state_dict()
    # pretrained = torch.load('./baseline/resnet34_lm_pt3_v2_129_4_10.0_wofc.pth')
    pretrained = torch.load(load_weight_path)
    pretrained_dict = pretrained['state_dict']

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    model.eval()
    print('==> Pretrained model is loaded.')

    # read image [RGB] and change scale to [0, 1]
    anchor = read_img(path='images/anchor.jpg')
    positive = read_img(path='images/positive.jpg')
    negative = read_img(path='images/negative.jpg')

    # transpose such that [1xCxHxW]
    anchor = torch.from_numpy(np.transpose(anchor, (2, 0, 1))[np.newaxis])
    positive = torch.from_numpy(np.transpose(positive, (2, 0, 1))[np.newaxis])
    negative = torch.from_numpy(np.transpose(negative, (2, 0, 1))[np.newaxis])

    # extract features
    anchor_fea = model(anchor.float().cuda()).cpu().detach().numpy()
    positive_fea = model(positive.float().cuda()).cpu().detach().numpy()
    negative_fea = model(negative.float().cuda()).cpu().detach().numpy()

    # calculate distance
    emb_size = 2048
    sim_a_and_p = 1 - pairwise_distances(np.reshape(anchor_fea, (1, emb_size)), np.reshape(positive_fea, (1, emb_size)), 'cosine')
    sim_a_and_n = 1 - pairwise_distances(np.reshape(anchor_fea, (1, emb_size)), np.reshape(negative_fea, (1, emb_size)), 'cosine')

    print('Similarity for [anchor, positive]: {:0.2f} \nSimilarity for [anchor, negative]: {:0.2f}'.format(
        sim_a_and_p[0][0], sim_a_and_n[0][0]))

    print("Ok")


if __name__ == '__main__':
    main()
