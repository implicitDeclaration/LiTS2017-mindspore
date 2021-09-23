# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""

from mindspore import Model
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager

from args import args
from utils.criterion import get_criterion
from utils.eval_utils import DiceCof
from utils.get_misc import get_dataset, \
    set_device, get_model, pretrained
from utils.optimizer import get_optimizer

set_seed(2)


def main(args):
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=args.enable_graph_kernel)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)

    # get model
    if args.arch != "UNet_3Plus":
        raise ValueError(f"Arch {args.arch} has not been supported yet.")

    net = get_model(args)

    if args.pretrained:
        pretrained(args, net)
    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)

    criterion = get_criterion(args)

    # save a yaml file to read to record parameters

    if args.is_dynamic_loss_scale:
        args.loss_scale = 1

    if args.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(args.loss_scale, drop_overflow_update=True)

    model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics={"dice": DiceCof()},
                  amp_level=args.amp_level, keep_batchnorm_fp32=args.keep_bn_fp32,
                  loss_scale_manager=loss_scale_manager)

    if args.evaluate:
        acc = model.eval(data.val_dataset)
        print(f"model's dice coff is {acc}")
        return

    config_ck = CheckpointConfig(save_checkpoint_steps=data.train_dataset.get_dataset_size(), keep_checkpoint_max=40)
    time_cb = TimeMonitor(data_size=data.train_dataset.get_dataset_size())
    ckpt_save_dir = "./ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix=args.arch + str(rank), directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()
    print("begin train")
    model.train(args.epochs, data.train_dataset, callbacks=[time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=True)

    print("train success")


if __name__ == '__main__':
    main(args)
