from __future__ import print_function, division, absolute_import

from bn_inception2 import bninception

import torch.nn as nn
import torch
from torch.nn import DataParallel
import torch.nn.functional as F


class Siamese(nn.Module):
    def __init__(self):

        super(Siamese, self).__init__()
        self.model = bninception()
        #self.norm = FeatureL2Norm()

    def forward(self, x1, x2):

        eA = self.model(x1)
        eB = self.model(x2)
        #fA = torch.sigmoid(fA)#self.norm(fA)
        #fB = torch.sigmoid(fB)#self.norm(fB)

        return eA, eB