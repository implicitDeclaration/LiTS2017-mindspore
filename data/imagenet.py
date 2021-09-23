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
"""
Data operations, will be used in train.py and eval.py
"""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision
import yaml
from mindspore.dataset.vision import ImageBatchFormat
from mindspore.dataset.vision.utils import Inter

from args import args


class ImageNet:
    def __init__(self, args):
        data_root = os.path.join(args.data, "imagenet")
        train_dir = os.path.join(data_root, "train")
        val_ir = os.path.join(data_root, "val")

        train_dir = "/old/dataset/ImageNet2012/ILSVRC2012_train"
        val_ir = "/old/dataset/ImageNet2012/ILSVRC2012_val"
        self.train_dataset = create_dataset_imagenet(train_dir, training=True)
        # self.train_loader = self.train_dataset.create_dict_iterator()
        self.val_dataset = create_dataset_imagenet(val_ir, training=False)
        # self.val_loader = self.val_dataset.create_dict_iterator()


def create_dataset_imagenet(dataset_dir, repeat_num=1, training=True):
    """
    create a train or eval imagenet2012 dataset for SwinTransformer

    Args:
        dataset_dir(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1

    Returns:
        dataset
    """

    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    device_num, rank_id = _get_rank_info()
    shuffle = True if training else False
    _, class_indexing = find_classes(dataset_dir)
    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers,
                                         shuffle=shuffle, class_indexing=class_indexing)
    else:
        data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers, shuffle=shuffle,
                                         num_shards=device_num, shard_id=rank_id, class_indexing=class_indexing)
    config_path = os.path.join("./configs/swin", f"{args.arch}.yaml")
    if os.path.exists(config_path):
        yaml_txt = open(config_path).read()
        # override args
        loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
        image_size = loaded_yaml["DATA"]["IMG_SIZE"] if "DATA" in loaded_yaml.keys() else 224
    else:
        image_size = 224

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    # define map operations
    # BICUBIC: 3
    if training:
        transform_img = \
            [
                vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                                              interpolation=Inter.BICUBIC),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
                vision.Normalize(mean=mean, std=std),
                vision.HWC2CHW(),
            ]
    else:
        # test transform complete
        transform_img = [
            vision.Decode(),
            vision.Resize(int(256 / 224 * image_size), interpolation=Inter.BICUBIC),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    transform_label = [C.TypeCast(mstype.int32)]

    data_set = data_set.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)
    data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)
    if args.mix_up and training:
        onehot_op = C.OneHot(num_classes=args.num_classes)
        data_set = data_set.map(operations=onehot_op, input_columns=["label"],
                                num_parallel_workers=args.num_parallel_workers)
    # apply batch operations
    data_set = data_set.batch(args.batch_size, drop_remainder=True, num_parallel_workers=args.num_parallel_workers)

    if args.mix_up and training:
        cutmix_batch_op = vision.CutMixBatch(ImageBatchFormat.NCHW, alpha=1.0, prob=1.0)
        data_set = data_set.map(operations=cutmix_batch_op, input_columns=["image", "label"],
                                num_parallel_workers=args.num_parallel_workers)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
