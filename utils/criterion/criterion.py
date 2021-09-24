from mindspore.nn.loss.loss import _Loss

from .bceLoss import BCELoss
from .diceloss import DiceLoss
from .iouLoss import IOULoss
from .msssimLoss import MSSSIM


class HybridLossSeg(_Loss):
    """CrossEntropy"""

    def __init__(self, args=None):
        super(HybridLossSeg, self).__init__()
        self.bce = BCELoss()
        self.iou = IOULoss()
        self.msss = MSSSIM(window_size=args.window_size, size_average=True, channel=args.in_channel)

    def construct(self, logit, label):
        loss2 = self.bce(logit, label) + self.iou(logit, label) + self.msss(logit, label)

        return loss2


def get_criterion(args):
    if args.loss_type.lower() == 'dice':
        criterion = DiceLoss()
    else:
        raise ValueError(f"{args.loss_type} loss has not been supported yet.")
    return criterion
