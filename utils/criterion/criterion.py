from mindspore.nn.loss.loss import _Loss

from .bceLoss import BCELoss
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
    if args.arch == 'UNet_3Plus':
        criterion = BCELoss()
    elif args.arch in ["UNet_3Plus_DeepSup", "UNet_3Plus_DeepSup_CGM"]:
        #TODO: Loss Function can not run in mindspore
        criterion = HybridLossSeg(args)
    else:
        raise ValueError(f"{args.arch}'s loss has not been supported yet.")
    return criterion
