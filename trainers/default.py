import time

import tqdm

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter

__all__ = ["train", "validate"]


def train(data, model, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    grad_norms = AverageMeter("GradNorms", ":6.2f")
    progress = ProgressMeter(
        data.train_dataset.get_dataset_size(),
        [batch_time, data_time, losses, grad_norms],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.set_train(True)

    batch_size = args.batch_size
    num_batches = data.train_dataset.get_dataset_size()
    end = time.time()
    for i, data in tqdm.tqdm(
            enumerate(data.train_loader), ascii=True, total=num_batches
    ):
        images = data["image"]
        target = data["label"]
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        grad_norm, loss = model(images, target)
        # measure accuracy and record loss
        losses.update(loss.asnumpy(), batch_size)
        grad_norms.update(grad_norm.asnumpy()[0], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return grad_norms.avg, losses.avg


def validate(data, model, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        data.val_dataset.get_dataset_size(), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.set_train(False)
    num_batches = data.val_dataset.get_dataset_size()
    batch_size = args.batch_size

    end = time.time()
    for i, data in tqdm.tqdm(
            enumerate(data.val_loader), ascii=True, total=num_batches
    ):
        # compute output
        images = data["image"]
        target = data["label"]

        output, loss = model.network(images, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    progress.display(data.val_dataset.get_dataset_size())

    if writer is not None:
        progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg
