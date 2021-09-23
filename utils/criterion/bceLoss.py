from mindspore import nn
from mindspore.nn.loss.loss import _Loss


class BCELoss(_Loss):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')

    def construct(self, logits, label):
        return self.bce_loss(logits, label)

# from mindspore import context
# from mindspore import Tensor
# from mindspore import dtype as mstype
# import numpy as np

# context.set_context(mode=context.PYNATIVE_MODE)
# model = BCELoss()
# data = Tensor(np.clip(np.random.randn(2, 3, 224, 224), 0, 1), mstype.float32)
#
# print(model(data, data))
