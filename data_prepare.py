import os

# attention
import SimpleITK as sitk
import numpy as np


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


data_dir = "./nii"

val_volumes = [f"volume-{i}.nii" for i in range(0, 28)]
val_segmentations = [f"segmentation-{i}.nii" for i in range(0, 28)]

val_dir = "test_slice"
makedirs(val_dir)
for volume, segmentation in zip(val_volumes, val_segmentations):
    ct = sitk.ReadImage(os.path.join(data_dir, volume), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    seg = sitk.ReadImage(os.path.join(data_dir, segmentation), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    assert ct_array.shape == seg_array.shape
    print(ct_array.shape, seg_array.shape)
    for slice in range(ct_array.shape[0] - 2):
        ct_array_dist = ct_array[slice:slice + 3]
        seg_array_dist = seg_array[slice:slice + 3]
        np.save(os.path.join(val_dir, volume.replace(".nii", f"{slice}.npy")), ct_array_dist)
        np.save(os.path.join(val_dir, segmentation.replace(".nii", f"{slice}.npy")), seg_array_dist)

train_volumes = [f"volume-{i}.nii" for i in range(28, 131)]
train_segmentations = [f"segmentation-{i}.nii" for i in range(28, 131)]
train_dir = "train_slice"
makedirs(train_dir)
for volume, segmentation in zip(train_volumes, train_segmentations):
    ct = sitk.ReadImage(os.path.join(data_dir, volume), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    seg = sitk.ReadImage(os.path.join(data_dir, segmentation), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    assert ct_array.shape == seg_array.shape
    print(ct_array.shape, seg_array.shape)
    for slice in range(ct_array.shape[0] - 2):
        ct_array_dist = ct_array[slice:slice + 3]
        seg_array_dist = seg_array[slice:slice + 3]
        if np.any(seg_array_dist):
            np.save(os.path.join(train_dir, volume.replace(".nii", f"{slice}.npy")), ct_array_dist)
            np.save(os.path.join(train_dir, segmentation.replace(".nii", f"{slice}.npy")), seg_array_dist)
