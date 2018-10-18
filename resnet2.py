from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


class ResNet(nn.Module):

    def __init__(self, pretrained, num_classes=751, num_features=256, dropout=0.5):
        super(ResNet, self).__init__()

        self.base = torchvision.models.resnet50(pretrained=pretrained)

        out_planes=self.base.fc.in_features
        self.conv1x1 = nn.Conv2d(out_planes, num_features, kernel_size=1, padding=0, bias=False)
        init.kaiming_normal_(self.conv1x1.weight, mode='fan_out')

        self.feat_bn2d = nn.BatchNorm2d(num_features)  # may not be used, not working on caffe
        init.constant_(self.feat_bn2d.weight, 1)  # initialize BN, may not be used
        init.constant_(self.feat_bn2d.bias, 0)    # iniitialize BN, may not be used

        self.fc_reid = nn.Linear(num_features, num_classes)
        init.normal_(self.fc_reid.weight, std=0.001)
        init.constant_(self.fc_reid.bias, 0)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, output_feature=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        # x = nn.AvgPool2d((8, 4), stride=1)(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        if not self.training:
            return x

        x = self.drop(x)
        x = self.conv1x1(x)
        x = self.feat_bn2d(x)
        x = F.relu(x)  # relu for bottlenck feature

        out = self.fc_reid(x.squeeze())

        return out

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet50(pretrained=True, num_classes=0):
    return ResNet(pretrained, num_classes)


