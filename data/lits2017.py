# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import os
import random

import cv2
import numpy as np
from mindspore import dataset as ds
from mindspore import dtype as mstype
from mindspore.dataset.transforms import c_transforms

from args import args

MIN_BOUND = -200.0
MAX_BOUND = 200.0
cv2.setNumThreads(0)


class LiTS2017:
    def __init__(self, args):
        data_root = os.path.join(args.data, "LiTS2017")
        train_dir = os.path.join(data_root, "train_slice")
        val_ir = os.path.join(data_root, "test_slice")

        self.train_dataset = create_dataset_lits2017(train_dir, training=True)
        # self.train_loader = self.train_dataset.create_dict_iterator()
        self.val_dataset = create_dataset_lits2017(val_ir, training=False)
        # self.val_loader = self.val_dataset.create_dict_iterator()


class MyIterable:
    def __init__(self, volume, seg):
        self._index = 0
        self._data = volume
        self._label = seg

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            data, label = np.load(self._data[self._index]), np.load(self._label[self._index])
            self._index += 1
            data = data.transpose(1, 2, 0)
            label = label.transpose(1, 2, 0)
            data = np.clip(data, a_min=MIN_BOUND, a_max=MAX_BOUND)
            data = 2 * (data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - 1
            label = np.clip(label, a_min=0, a_max=1)
            return (data, label)

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)


def random_resize_img_label(img, label):
    img_size_w = int(np.random.randint(img.shape[0], img.shape[0] * 1.5, 1))
    img_size_h = int(np.random.randint(img.shape[1], img.shape[1] * 1.5, 1))
    img2 = cv2.resize(img, (img_size_w, img_size_h), interpolation=cv2.INTER_NEAREST)
    label2 = cv2.resize(label, (img_size_w, img_size_h))
    th, label2 = cv2.threshold(src=label2, thresh=0.5, maxval=1)
    return img2, label2


def random_crop_img_label(img, label):
    begin_x = random.randint(0, img.shape[0] - args.crop_size)
    begin_y = random.randint(0, img.shape[1] - args.crop_size)

    img = img[begin_x:begin_x + args.crop_size, begin_y:begin_y + args.crop_size].copy()
    label = label[begin_x:begin_x + args.crop_size, begin_y:begin_y + args.crop_size].copy()
    return img, label


def flip_img_label(img, mask):
    h_flip = np.random.random()
    if h_flip > 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)
    v_flip = np.random.random()
    if v_flip > 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    return img, mask


def center_crop(img, label):
    begin_x = (img.shape[0] - args.crop_size) // 2
    begin_y = (img.shape[1] - args.crop_size) // 2

    img1 = img[begin_x:begin_x + args.crop_size, begin_y:begin_y + args.crop_size].copy()
    label1 = label[begin_x:begin_x + args.crop_size, begin_y:begin_y + args.crop_size].copy()
    return img1, label1


def hwc2chw(img, label):
    return img.transpose(2, 0, 1).copy(), label.transpose(2, 0, 1).copy()


def create_dataset_lits2017(path, repeat_num=1, training=False):
    volumes = []
    segmentations = []
    for dir in os.listdir(path):
        if "volume" in dir and dir.endswith(".npy"):
            volumes.append(os.path.join(path, dir))
    for volume in volumes:
        segmentations.append(volume.replace("volume", "segmentation"))

    device_num, rank_id = _get_rank_info()
    shuffle = True if training else False

    column_names = ["volume", "segmentation"]
    my_iter = MyIterable(volumes, segmentations)
    if device_num == 1:
        data_set = ds.GeneratorDataset(source=my_iter,
                                       num_parallel_workers=args.num_parallel_workers,
                                       column_names=column_names,
                                       shuffle=shuffle)
    else:
        data_set = ds.GeneratorDataset(source=my_iter,
                                       num_parallel_workers=args.num_parallel_workers,
                                       column_names=column_names,
                                       shuffle=shuffle,
                                       num_shards=device_num,
                                       shard_id=rank_id)
    if training:
        transform_img = [
            flip_img_label,
            random_crop_img_label,
            hwc2chw,
        ]
    else:
        transform_img = [
            center_crop,
            hwc2chw,
        ]

    transform_label = [c_transforms.TypeCast(mstype.float32)]
    data_set = data_set.map(input_columns=["volume", "segmentation"],
                            num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)

    data_set = data_set.map(input_columns="segmentation",
                            num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)

    data_set = data_set.batch(args.batch_size, drop_remainder=False)
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
