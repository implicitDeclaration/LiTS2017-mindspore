from mindspore import ops
from mindspore.nn.loss.loss import _Loss


class IOULoss(_Loss):
    def __init__(self):
        super(IOULoss, self).__init__()
        self.sum_op = ops.ReduceSum(keep_dims=False)
        self.mean_op = ops.ReduceMean(keep_dims=False)

    def construct(self, pred, target):
        Iand1 = self.sum_op(target * pred, (1, 2, 3))
        Ior1 = self.sum_op(target, (1, 2, 3)) + self.sum_op(pred, (1, 2, 3)) - Iand1
        IoU1 = 1 - self.mean_op(Iand1 / Ior1)
        return IoU1
