import mindspore.nn as nn
from mindspore import ParameterTuple
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import ops
from mindspore.communication.management import get_group_size
from mindspore.ops import clip_by_global_norm
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._auto_parallel_context import auto_parallel_context

scale = Tensor(1., dtype=mstype.float32)

grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor")
def _grad_scale(x):
    div = ops.Div()
    return div(x, scale)


class ClipGlobalNorm(nn.Cell):
    def __init__(self, clip_norm):
        super(ClipGlobalNorm, self).__init__()
        self.clip_norm = clip_norm

    def construct(self, grad):
        return clip_by_global_norm(grad, self.clip_norm)


class WithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label):
        out = self._backbone(data)
        if self.training:
            return self._loss_fn(out, label)
        return out, self._loss_fn(out, label)


# form dsj
GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = .5

get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor")
def _get_square_sum(x):
    norm = P.ReduceSum(False)(F.square(x), ())
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


class TrainingWrapper(nn.Cell):
    """TrainingWrapper"""

    def __init__(self, network, optimizer, max_norm=2.):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.params = ParameterTuple(optimizer.parameters)
        self.optimizer = optimizer
        self.max_norm = max_norm
        self.grad = C.GradOperation(get_by_list=True, sens_param=False)
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        class_list = [context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL]
        if self.parallel_mode in class_list:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        # from dsj
        self.hyper_map = ops.HyperMap()
        self.clip_norm = ClipGlobalNorm(max_norm=max_norm)

    def construct(self, *args):
        """construct"""

        loss = self.network(*args)
        grads = self.grad(self.network, self.params)(*args)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        grads = self.hyper_map(grad_scale, grads)
        grads = self.clip_norm(grads)
        square_sum = self.hyper_map(get_square_sum, grads)
        global_norm = F.sqrt(F.addn(square_sum))
        return global_norm, F.depend(loss, self.optimizer(grads))

# from mindspore import Tensor
# from mindspore import dtype as mstype
# class GetNorm(nn.Cell):
#     def __init__(self):
#         super(GetNorm, self).__init__()
#
#     def construct(self, weights):
#         norm = []
#         for weight in weights:
#             # print(weight.shape)
#             norm.append(np.norm(weight))
#         return np.norm(norm).asnumpy()
