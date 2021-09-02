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
import copy
import pathlib
import time

from mindspore import context
from mindspore.common import set_seed
from ruamel import yaml
from tensorboardX import SummaryWriter

from args import args
from utils.cell import WithLossCell, TrainingWrapper
from utils.critetion import get_criterion
from utils.get_misc import get_dataset, \
    set_device, get_trainer, \
    get_model, pretrained, save_checkpoint, get_directories, \
    resume
from utils.logging import AverageMeter, ProgressMeter
from utils.optimizer import get_lr, get_optimizer
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
    args_text = copy.copy(args.__dict__)
    del args_text['ckpt_base_dir']
    with open(run_base_dir / 'args.yaml', 'w', encoding="utf-8") as f:
        yaml.dump(
            args_text,
            f,
            Dumper=yaml.RoundTripDumper,
            default_flow_style=False,
            allow_unicode=True,
            indent=4
        )

    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        model,
        append_dict={
            "epoch": 0,
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            # "optimizer": optimizer,
            "curr_acc1": 0.,
        },
        is_best=False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )
    # Start training

    model = WithLossCell(model, criterion)
    model = TrainingWrapper(model, optimizer, max_norm=args.clip_grad)

    for epoch in range(args.start_epoch, args.epochs):

        lr_policy(epoch, iteration=None)

        cur_lr = get_lr(model.optimizer)
        print(f"==> current_learning_rate: {cur_lr}")
        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data, model, epoch, args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data, model, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")
            save_checkpoint(
                model,
                append_dict={
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    # "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best=is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

            epoch_time.update((time.time() - end_epoch) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(
                writer, prefix="diagnostics", global_step=epoch
            )

            writer.add_scalar("test/lr", cur_lr, epoch)
            end_epoch = time.time()

    # from mindspore import Model
    #
    #
    # from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
    # if args.is_dynamic_loss_scale:
    #     args.loss_scale = 1
    # from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
    #
    # if args.is_dynamic_loss_scale == 1:
    #     loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    # else:
    #     loss_scale_manager = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    #
    # model = Model(model, loss_fn=criterion, optimizer=optimizer, metrics={'acc'},
    #               amp_level="O3", keep_batchnorm_fp32=True, loss_scale_manager=loss_scale_manager)
    #
    # config_ck = CheckpointConfig(save_checkpoint_steps=10007, keep_checkpoint_max=40)
    # time_cb = TimeMonitor(data_size=10007)
    # ckpt_save_dir = "./ckpt_" + str(rank) + "/"
    # ckpoint_cb = ModelCheckpoint(prefix="swin_transformer", directory=ckpt_save_dir,
    #                              config=config_ck)
    # loss_cb = LossMonitor()
    # model.train(args.epochs, data.train_dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
    # print("train success")


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Set, "
            "Batch Size, "
            "Base Config, "
            "Name, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5, "
            "Arch, "
            "Trainer, "
            "Optimizer, "
            "WeightDecay, "
            "Run Base Dir\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            ("{now}, "
             "{set}, "
             "{batch_size}, "
             "{base_config}, "
             "{name}, "
             "{curr_acc1:.02f}, "
             "{curr_acc5:.02f}, "
             "{best_acc1:.02f}, "
             "{best_acc5:.02f}, "
             "{best_train_acc1:.02f}, "
             "{best_train_acc5:.02f}, "
             "{arch}, "
             "{trainer}, "
             "{optimizer}, "
             "{weight_decay}, "
             "{run_base_dir}\n"
             ).format(now=now, **kwargs)
        )


if __name__ == '__main__':
    main(args)
