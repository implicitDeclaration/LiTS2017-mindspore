from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops
from mindspore.nn.loss.loss import _Loss


class DiceLoss(_Loss):
    """CrossEntropy"""

    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = Tensor(1., dtype=mstype.float32)

    def construct(self, logit, label):
        intersection = ops.ReduceSum()(logit * label, (1, 2, 3))
        union = ops.ReduceSum()(logit, (1, 2, 3)) + \
                ops.ReduceSum()(label, (1, 2, 3))
        mean_loss = ops.ReduceMean()((2 * intersection + self.smooth) / (union + self.smooth))

        return 1 - mean_loss
