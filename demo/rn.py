'''
    Network structure with a stride of 1, an image size of 384x128

    @author: Elyor Kodirov
    @date: 19/02/2019
'''

from __future__ import division

import torch
import torchvision
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input,axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class ResNet50_pytorch(nn.Module):

    def __init__(self, pretrained=False, stride=1):
        super(ResNet50_pytorch, self).__init__()
        self.base = torchvision.models.resnet50(pretrained=pretrained)
        if stride == 1:
            for mo in self.base.layer4[0].modules():
                if isinstance(mo, nn.Conv2d):
                    mo.stride = (1, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # nn.AvgPool2d(7)
        self.feat_bn = nn.BatchNorm2d(2048)  # may not be used, not working on caffe


    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x = self.avgpool(x)
        x = self.feat_bn(x).squeeze()

        return x


