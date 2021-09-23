# -*- coding: utf-8 -*-
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.common import initializer
from mindspore.common.initializer import HeNormal, Constant
from scipy.stats import truncnorm


class TruncatedNormal(initializer.Initializer):
    """
    Initialize a truncated normal distribution which is a bounded normal distribution within N(low, high).

    Args:
        sigma (float): The sigma of the array. Default: 0.01.

    Returns:
        Array, truncated normal array.
    """

    def __init__(self, sigma=0.01):
        super(TruncatedNormal, self).__init__()
        self.sigma = sigma

    def _initialize(self, arr):
        tmp = Tensor(truncnorm.rvs(-2, 2, loc=0, scale=self.sigma, size=arr.shape, random_state=None), mstype.float32)
        initializer._assignment(arr, tmp)


trunc_normal_ = TruncatedNormal(sigma=0.02)
he_normal_ = HeNormal(mode='fan_in')
constant0_ = Constant(value=0)
constant1_ = Constant(value=1)


class ConvBNReLU(nn.Cell):
    def __init__(self, in_size, out_size, ks,
                 stride, has_bias, is_bn=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=ks,
                              stride=stride, has_bias=has_bias)
        self.bn = None
        if is_bn:
            self.bn = nn.BatchNorm2d(num_features=out_size, momentum=0.9)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)

        return x


def unetConv2(in_size, out_size, is_batchnorm, n=2, ks=3, stride=1):
    cells_list = []
    for _ in range(1, n + 1):
        cell = ConvBNReLU(in_size, out_size, ks, stride, has_bias=True, is_bn=is_batchnorm)
        cells_list.append(cell)
        in_size = out_size
    SequentialCell = nn.SequentialCell(cells_list)
    return SequentialCell


class unetUp(nn.Cell):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(out_size * 2, out_size, False)
        if is_deconv:
            self.up = nn.Conv2dTranspose(in_size, out_size, kernel_size=4,
                                         stride=2, padding=1, pad_mode='pad', has_bias=True)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        self._init_weights()

    def _init_weights(self):
        for _, m in self.cells_and_names():
            if isinstance(m, unetConv2):
                he_normal_._initialize(m.weight)
                if isinstance(m, nn.Dense) and m.bias is not None:
                    constant0_(m.bias)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2dTranspose):
                he_normal_._initialize(m.weight)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    constant0_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                trunc_normal_._initialize(m.gamma)
                constant0_._initialize(m.beta)

    def construct(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = P.Concat(1)([outputs0, input[i]])
        return self.conv(outputs0)


class unetUp_origin(nn.Cell):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        self._init_weights()

    def _init_weights(self):
        for _, m in self.cells_and_names():
            if isinstance(m, unetConv2):
                he_normal_._initialize(m.weight)
                if isinstance(m, nn.Dense) and m.bias is not None:
                    constant0_(m.bias)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2dTranspose):
                he_normal_._initialize(m.weight)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    constant0_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                trunc_normal_._initialize(m.gamma)
                constant0_._initialize(m.beta)

    def construct(self, inputs0, *input):

        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = P.Concat(1)([outputs0, input[i]])
        return self.conv(outputs0)


class GlobalAveragePool(nn.Cell):
    def __init__(self):
        super(GlobalAveragePool, self).__init__()
        self.ops = P.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.ops(x, (2, 3))
        return x


class UpSample(nn.Cell):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(UpSample, self).__init__()

        self.ops = nn.ResizeBilinear()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def construct(self, x):
        x = self.ops(x, scale_factor=self.scale_factor, align_corners=self.align_corners)
        return x
