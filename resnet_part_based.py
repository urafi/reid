from __future__ import absolute_import
import torch
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

        self.num_stripes_horizontal = 4
        self.num_stripes_vertical = 1

        total_stripes = self.num_stripes_horizontal * self.num_stripes_vertical

        self.local_conv_list = nn.ModuleList()
        for _ in range(total_stripes):
            self.local_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, num_features, 1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True)
            ))

        self.fc_list = nn.ModuleList()
        for _ in range(total_stripes):
                fc = nn.Linear(num_features, num_classes)
                init.normal(fc.weight, std=0.001)
                init.constant(fc.bias, 0)
                self.fc_list.append(fc)

    def forward(self, feat, output_feature=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            feat = module(feat)

        stripe_h = int(feat.size(2) / self.num_stripes_horizontal)
        stripe_v = int(feat.size(3) / self.num_stripes_vertical)
        local_feat_list = []
        logits_list = []
        for i in range(self.num_stripes_horizontal):
            # shape [N, C, 1, 1]
            for j in range(self.num_stripes_vertical):
                local_feat = F.avg_pool2d(
                    feat[:, :, i * stripe_h: (i + 1) * stripe_h, j * stripe_v: (j + 1) * stripe_v],
                    (stripe_h, stripe_v))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list[i](local_feat)
                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list.append(local_feat)
                logits_list.append(self.fc_list[i](local_feat))

        if not self.training:

            f = torch.cat(local_feat_list, 1)
            return f



        return logits_list

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
