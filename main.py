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

from args import args
from utils.critetion import get_criterion
from utils.get_misc import get_dataset, \
    set_device, get_trainer, \
    get_model, pretrained, get_directories, \
    resume
from utils.optimizer import get_optimizer
from utils.schedulers import get_policy

set_seed(2)


def main(args):
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=True)
    rank = set_device(args)
    train, validate = get_trainer(args)

    # get model
    model = get_model(args)
    if args.pretrained:
        pretrained(args, model)

    optimizer = get_optimizer(args, model)
    data = get_dataset(args)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    criterion = get_criterion(args)

    if args.resume:
        resume(args, model, optimizer)

    # Data loading code
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        return

    loss_scale_manager = None

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        best_acc1 = resume(args, model, optimizer)

    # Data loading code
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )

        return

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

    # save a yaml file to read to record parameters

    from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
    if args.is_dynamic_loss_scale:
        args.loss_scale = 1
    from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager

    if args.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)

    model = Model(model, loss_fn=criterion, optimizer=optimizer, metrics={'acc'},
                  amp_level="O3", keep_batchnorm_fp32=True, loss_scale_manager=loss_scale_manager)

    config_ck = CheckpointConfig(save_checkpoint_steps=10007, keep_checkpoint_max=40)
    time_cb = TimeMonitor(data_size=10007)
    ckpt_save_dir = "./ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="swin_transformer", directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()
    model.train(args.epochs, data.train_dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
    print("train success")


if __name__ == '__main__':
    main(args)
