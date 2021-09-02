import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.nn import LossBase
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.cast = ops.Cast()

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss2 = self.ce(logit, label)
        return loss2


def get_criterion(args):
    assert args.label_smoothing >= 0. and args.label_smoothing <= 1.

    if args.mix_up > 0.:
        # smoothing is handled with mixup label transform
        criterion = CrossEntropySmooth(sparse=False, reduction="mean", num_classes=args.num_classes)
    elif args.label_smoothing > 0.:
        criterion = CrossEntropySmooth(sparse=True, reduction="mean",
                                       smooth_factor=args.label_smoothing, num_classes=args.num_classes)
    else:
        criterion = CrossEntropySmooth(sparse=True, reduction="mean", num_classes=args.num_classes)

    return criterion
