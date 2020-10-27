import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair


class WeightDropConv2d(nn.Conv2d):
    """
    Reimplementing baal version of WeightDropConv2d, because it doesn't support multi-GPU training
    For details, see https://github.com/pytorch/pytorch/issues/8637
    """
    def __init__(self, weight_dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_dropout = weight_dropout

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dropped_weight = torch.nn.functional.dropout(self.weight, p=self.weight_dropout, training=True)
        if self.bias is not None:
            dropped_bias = torch.nn.functional.dropout(self.bias, p=self.weight_dropout, training=True)
            return self._conv_forward(input, dropped_weight, dropped_bias)
        else:
            return self._conv_forward(input, dropped_weight, self.bias)

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class WeightDropLinear(nn.Linear):
    """
    Reimplementing baal version of WeightDropLinear, because it doesn't support multi-GPU training
    For details, see https://github.com/pytorch/pytorch/issues/8637
    """
    def __init__(self, weight_dropout=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_dropout = weight_dropout

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dropped_weight = torch.nn.functional.dropout(self.weight, p=self.weight_dropout, training=True)
        dropped_bias = torch.nn.functional.dropout(self.bias, p=self.weight_dropout, training=True)
        return F.linear(input, dropped_weight, dropped_bias)


def uniform_dropout(input, p=0.5, training=False, inplace=False):
    beta = p
    assert 0.0 < beta <= 1.0
    if training:
        out = input * (1.0 + torch.empty(input.shape, device=input.device).uniform_(-beta, beta))
        if inplace:
            raise NotImplementedError()
        return out
    else:
        return input