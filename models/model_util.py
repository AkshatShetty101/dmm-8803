import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Union, Tuple

from PointCNN.core.util_funcs import UFloatTensor

def EndChannels(f, make_contiguous = False):
    """ Class decorator to apply 2D convolution along end channels. """

    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, x):
            x = x.permute(0,3,1,2)
            x = self.f(x)
            x = x.permute(0,2,3,1)
            return x

    return WrappedLayer()

class Dense(nn.Module):
    """
    Single layer perceptron with optional activation, batch normalization, and dropout.
    """

    def __init__(self, in_features : int, out_features : int,
                 drop_rate : int = 0, with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ELU()
                ) -> None:
        """
        :param in_features: Length of input featuers (last dimension).
        :param out_features: Length of output features (last dimension).
        :param drop_rate: Drop rate to be applied after activation.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Dense, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        # self.bn = LayerNorm(out_channels) if with_bn else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Linear.
        :return: Tensor with linear layer and optional activation, batchnorm,
        and dropout applied.
        """
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        # if self.bn:
        #     x = self.bn(x)
        if self.drop:
            x = self.drop(x)
        return x

class Conv(nn.Module):
    """
    2D convolutional layer with optional activation and batch normalization.
    """

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : Union[int, Tuple[int, int]], with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ELU()
                ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias = not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.9) if with_bn else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with convolutional layer and optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class SepConv(nn.Module):
    """ Depthwise separable convolution with optional activation and batch normalization"""

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : Union[int, Tuple[int, int]],
                 depth_multiplier : int = 1, with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ELU()
                 ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :depth_multiplier: Depth multiplier for middle part of separable convolution.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(SepConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups = in_channels),
            nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, bias = not with_bn)
        )

        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.9) if with_bn else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class LayerNorm(nn.Module):
    """
    Batch Normalization over ONLY the mini-batch layer (suitable for nn.Linear layers).
    """

    def __init__(self, N : int, dim : int, *args, **kwargs) -> None:
        """
        :param N: Batch size.
        :param D: Dimensions.
        """
        super(LayerNorm, self).__init__()
        if dim == 1:
            self.bn = nn.BatchNorm1d(N, *args, **kwargs)
        elif dim == 2:
            self.bn = nn.BatchNorm2d(N, *args, **kwargs)
        elif dim == 3:
            self.bn = nn.BatchNorm3d(N, *args, **kwargs)
        else:
            raise ValueError("Dimensionality %i not supported" % dim)

        self.forward = lambda x: self.bn(x.unsqueeze(0)).squeeze(0)


def huber_loss(error, delta, weight=None):
    delta = torch.ones_like(error) * delta
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    # Note condisder batch mean
    if weight is not None:
        losses *= weight

    return losses.mean()


def smooth_l1_loss(input, target, sigma=1.0, size_average=True):
    '''
    input: B, *
    target: B, *

    '''
    # smooth_l1_loss with sigma
    """
            (sigma * x)^2/2  if x<1/sigma^2
    f(x)=
            |x| - 1/(2*sigma^2) otherwise
    """
    assert input.shape == target.shape

    diff = torch.abs(input - target)

    mask = (diff < (1. / sigma**2)).detach().type_as(diff)

    output = mask * torch.pow(sigma * diff, 2) / 2.0 + (1 - mask) * (diff - 1.0 / (2.0 * sigma**2.0))
    loss = output.sum()
    if size_average:
        loss = loss / input.shape[0]

    return loss


def get_box3d_corners_helper(centers, headings, sizes):

    N = centers.shape[0]
    l = sizes[:, 0]  # (N)
    w = sizes[:, 1]  # (N)
    h = sizes[:, 2]  # (N)
    x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], 1)  # (N,8)
    y_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], 1)  # (N,8)
    z_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], 1)  # (N,8)
    corners = torch.stack([x_corners, y_corners, z_corners], 1)  # (N,3,8)

    c = torch.cos(headings)
    s = torch.sin(headings)
    ones = headings.new_ones(N)
    zeros = headings.new_zeros(N)
    row1 = torch.stack([c, zeros, s], 1)  # (N,3)
    row2 = torch.stack([zeros, ones, zeros], 1)
    row3 = torch.stack([-s, zeros, c], 1)
    R = torch.stack([row1, row2, row3], 1)  # (N,3,3)

    # (N,3,3) * ((N,3,8))
    corners_3d = torch.bmm(R, corners)  # (N,3,8)
    corners_3d = corners_3d + centers.unsqueeze(2)  # (N,3,8)
    corners_3d = torch.transpose(corners_3d, 1, 2).contiguous()  # (N,8,3)
    return corners_3d
