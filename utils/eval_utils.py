import numpy as np
from mindspore.nn.metrics import Metric


def _dicecof(pred, target):
    smooth = 1.
    num = pred.shape[0]
    m1 = np.reshape(pred, (num, -1))  # Flatten
    m2 = np.reshape(target, (num, -1))  # Flatten
    intersection = np.sum(m1 * m2, axis=1, keepdims=False)
    m1 = np.sum(m1, axis=1, keepdims=False)
    m2 = np.sum(m2, axis=1, keepdims=False)
    return (2. * intersection + smooth) / (m1 + m2 + smooth)


class DiceCof(Metric):
    def __init__(self):
        super(DiceCof, self).__init__()

        self._correct_num = 0
        self._samples_num = 0
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self._correct_num = 0
        self._samples_num = 0

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        cof = _dicecof(y_pred, y)
        self._correct_num += np.sum(cof)
        self._samples_num += y_pred.shape[0]

    def eval(self):
        """
        Computes the top-k categorical accuracy.

        Returns:
            Float, computed result.
        """
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._correct_num / self._samples_num
