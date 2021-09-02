from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import AdamWeightDecay
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.optim.optimizer import _grad_scale
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.composite.clip_ops import get_square_sum


def get_lr(optimizer):
    return optimizer.learning_rate.asnumpy()


def get_optimizer(args, model):
    # if args.is_dynamic_loss_scale:
    #     args.loss_scale = 1
    #
    # if args.is_dynamic_loss_scale == 1:
    #     loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    # else:
    #     loss_scale_manager = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    optim_type = args.optimizer.lower()
    params = get_param_groups(model)

    if optim_type == "momentum":
        optim = Momentum(
            params=params,
            learning_rate=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif optim_type == "adamw":
        optim = AdamWeightDecay(
            params=params,
            learning_rate=args.lr,
            beta1=args.beta[0],
            beta2=args.beta[1],
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    elif optim_type == "adamwself":
        optim = AdamWeightDecaySelf(
            clip_norm=args.clip_norm,
            params=params,
            learning_rate=args.lr,
            beta1=args.beta[0],
            beta2=args.beta[1],
            eps=args.eps,
            weight_decay=args.weight_decay
        )
    elif optim_type == "momentumself":
        optim = MomentumSelf(
            clip_norm=args.clip_norm,
            params=params,
            learning_rate=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"optimizer {optim_type} is not supported")

    return optim


def get_param_groups(network):
    """ get param groups """
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


class AdamWeightDecaySelf(AdamWeightDecay):
    def __init__(self, clip_norm, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecaySelf, self).__init__(params, learning_rate, beta1, beta2, eps, weight_decay)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.scale = Tensor([1.], mstype.float32)
        self.hyper_map_clip_norm = C.HyperMap()
        self.greater_equal_clip_norm = P.GreaterEqual()

    def construct(self, gradients):
        square_sum = self.hyper_map_clip_norm(get_square_sum, gradients)
        global_norm = F.sqrt(F.addn(square_sum))
        cond = self.greater_equal_clip_norm(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        reciprocal_scale = self.clip_norm / global_norm
        # gradients = self.clip_grad_norm(gradients, self.clip_norm)
        # TODO: clip grad norm这步有问题
        # gradients = clip_grad(gradients, reciprocal_scale)
        gradients = self.clip_grad_norm(gradients, reciprocal_scale)
        result = super(AdamWeightDecaySelf, self).construct(gradients)
        return result

    def clip_grad_norm(self, gradients, reciprocal_scale):
        gradients = self.map_(F.partial(_grad_scale, reciprocal_scale), gradients)
        return gradients


class MomentumSelf(Momentum):
    def __init__(self, clip_norm, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0,
                 use_nesterov=False):
        super(MomentumSelf, self).__init__(params, learning_rate, momentum, weight_decay, loss_scale, use_nesterov)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.reciprocal_scale_clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map_clip_norm = C.HyperMap()
        self.greater_equal_clip_norm = P.GreaterEqual()
        self.div_clip_norm = P.Div()
        self.assign_replace = P.Assign()

    def construct(self, gradients):
        square_sum = self.hyper_map_clip_norm(get_square_sum, gradients)
        global_norm = F.sqrt(F.addn(square_sum))
        cond = self.greater_equal_clip_norm(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        self.assign_replace(self.reciprocal_scale_clip_norm, self.clip_norm / global_norm)
        # gradients = self.clip_grad_norm(gradients, self.clip_norm)
        # TODO: clip grad norm这步有问题
        # gradients = clip_grad(gradients, reciprocal_scale)
        gradients = self.clip_grad_norm(gradients)
        result = super(MomentumSelf, self).construct(gradients)
        return result

    def clip_grad_norm(self, gradients):
        gradients = self.map_(F.partial(_grad_clip_norm, self.reciprocal_scale_clip_norm), gradients)
        return gradients


op_mul = P.Mul()
_grad_clip_norm = C.MultitypeFuncGraph("grad_clip_norm")


@_grad_clip_norm.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return op_mul(grad, F.cast(scale, F.dtype(grad)))


@_grad_clip_norm.register("Tensor", "Tensor")
def tensor_grad_scale_with_tensor(scale, grad):
    """Get grad with scale."""
    return op_mul(grad, F.cast(scale, F.dtype(grad)))

# from models.swintransformer.swin_transformer import SwinTransformer
# model =SwinTransformer()
# from mindspore import context
# context.set_context(mode=context.PYNATIVE_MODE)
# opt = AdamWeightDecaySelf(clip_norm=1, params=model.trainable_params())
# opt(model.trainable_params())
