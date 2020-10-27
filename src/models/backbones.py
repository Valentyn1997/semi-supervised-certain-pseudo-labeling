import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dropouts import WeightDropLinear, WeightDropConv2d

logger = logging.getLogger(__name__)


# Credits to https://github.com/kekmodel/FixMatch-pytorch/blob/master/models/wideresnet.py


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, weight_dropout=0.0, dropout_method=F.dropout,
                 activate_before_residual=False):
        super(BasicBlock, self).__init__()
        if weight_dropout == 0.0:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = WeightDropConv2d(weight_dropout, in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                          bias=False)
            self.conv2 = WeightDropConv2d(weight_dropout, out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = PSBatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.bn2 = PSBatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.dropout_method = dropout_method
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = self.dropout_method(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, weight_dropout=0.0,
                 dropout_method=F.dropout, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate, weight_dropout, dropout_method,
                                      activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, weight_dropout, dropout_method,
                    activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, weight_dropout, dropout_method, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0, weight_dropout=0.0, dropout_method=F.dropout,
                 after_bn_drop_rate=0.0):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, channels[0], channels[1], block, 1, drop_rate, weight_dropout, dropout_method,
                                   activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, channels[1], channels[2], block, 2, drop_rate, weight_dropout, dropout_method)
        # 3rd block
        self.block3 = NetworkBlock(n, channels[2], channels[3], block, 2, drop_rate, weight_dropout, dropout_method)
        # global average pooling and classifier
        self.bn1 = PSBatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if weight_dropout == 0.0:
            self.fc = nn.Linear(channels[3], num_classes)
        else:
            self.fc = WeightDropLinear(weight_dropout, channels[3], num_classes)
        self.channels = channels[3]
        self.after_bn_drop_rate = after_bn_drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, WeightDropConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, PSBatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear) or isinstance(m, WeightDropLinear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        if self.after_bn_drop_rate > 0.0:
            out = F.dropout(out, p=self.after_bn_drop_rate, training=self.training)
        return self.fc(out)
