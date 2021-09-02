import collections.abc
from itertools import repeat

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import numpy
from mindspore import ops
from mindspore.common import initializer
from scipy.stats import truncnorm


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + Tensor(numpy.random.randn(*shape), dtype=mstype.float32)
    random_tensor = floor(random_tensor)  # binarize
    output = div(x, keep_prob) * random_tensor

    return output


class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, x):
        return x


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TruncatedNormal(initializer.Initializer):
    """
    Initialize a truncated normal distribution which is a bounded normal distribution within N(low, high).

    Args:
        sigma (float): The sigma of the array. Default: 0.01.

    Returns:
        Array, truncated normal array.
    """

    def __init__(self, sigma=0.01):
        super(TruncatedNormal, self).__init__(sigma=sigma)
        self.sigma = sigma

    def _initialize(self, arr):
        tmp = Tensor(truncnorm.rvs(-2, 2, loc=0, scale=self.sigma, size=arr.shape, random_state=None), mstype.float32)
        initializer._assignment(arr, tmp)


floor = ops.Floor()
div = ops.Div()

to_2tuple = _ntuple(2)

reshape = ops.Reshape()
matmul = ops.BatchMatMul()
transpose = ops.Transpose()
expand_dims = ops.ExpandDims()
trunc_normal_ = TruncatedNormal(sigma=0.02)
constant0_ = initializer.Constant(value=0)
constant1_ = initializer.Constant(value=1)
