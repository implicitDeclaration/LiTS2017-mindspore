import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.nn.loss.loss import _Loss

expand_dims = ops.ExpandDims()


def gaussian(window_size, sigma):
    gauss = Tensor(np.array([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]),
                   dtype=mstype.float32)
    return gauss / ops.ReduceSum(keep_dims=False)(gauss)


def create_window(window_size, channel=1):
    _1D_window = ops.ExpandDims()(gaussian(window_size, 1.5), 1)
    _2D_window = ops.MatMul(transpose_b=True)(_1D_window, _1D_window)
    # _2D_window = _1D_window.mm(_1D_window.t()).float()
    _2D_window = expand_dims(_2D_window, 0)
    _2D_window = expand_dims(_2D_window, 0)
    window = ops.repeat_elements(_2D_window, rep=channel, axis=0)
    # window = _2D_window.expand(channel, 1, window_size, window_size)
    return window


def ssim(img1, img2, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if ops.ReduceMax(keep_dims=False)(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if ops.ReduceMin(keep_dims=False)(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
    padd = 0
    (_, channel, height, width) = img1.shape
    conv_op0 = ops.Conv2D(out_channel=window.shape[0],
                          kernel_size=window.shape[-1],
                          pad_mode='pad',
                          pad=padd,
                          group=channel,
                          mode=1)
    mu1 = conv_op0(img1, window)
    mu2 = conv_op0(img2, window)

    mu1_sq = ops.Pow()(mu1, 2)
    mu2_sq = ops.Pow()(mu2, 2)
    mu1_mu2 = mu1 * mu2
    conv_op1 = ops.Conv2D(out_channel=window.shape[0],
                          kernel_size=window.shape[-1],
                          pad_mode='pad',
                          pad=padd,
                          group=channel,
                          mode=1)
    sigma1_sq = conv_op1(img1 * img1, window) - mu1_sq
    sigma2_sq = conv_op1(img2 * img2, window) - mu2_sq
    sigma12 = conv_op1(img1 * img2, window) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = ops.ReduceMean(False)(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ops.ReduceMean(keep_dims=False)(ssim_map)
    else:
        ret = ops.ReduceMean(keep_dims=False)(ssim_map, (1, 2, 3))
    if full:
        return ret, cs
    return ret


def msssim(img1, img2, weights, window, size_average=True, val_range=None, normalize=False):
    levels = weights.shape[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window, size_average=size_average, full=True,
                       val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = nn.AvgPool2d(kernel_size=2)(img1)
        img2 = nn.AvgPool2d(kernel_size=2)(img2)

    mssim = ops.Stack()(mssim)
    mcs = ops.Stack()(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = ops.Pow()(mcs, weights)
    pow2 = ops.Pow()(mssim, weights)
    output = ops.ReduceProd(keep_dims=False)(pow1[:-1] * pow2[-1])
    return output


class MSSSIM(_Loss):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.weights = Tensor(np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]), dtype=mstype.float32)
        self.window = create_window(window_size, channel=channel)
        self.window = ops.Cast()(self.window, mstype.float32)

    def construct(self, img1, img2):
        # TODO: store window between calls if possible,
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return msssim(img1, img2, self.weights, window=self.window,
                      size_average=self.size_average, normalize=True)
