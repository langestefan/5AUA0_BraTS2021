import numpy as np
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandRotated,
    RandZoomd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandFlipd,
    ToTensord,
)


def brats_train_transform(image_keys, all_keys, spatial_size=(240, 240)):
    
    train_transform = Compose([
        AddChanneld(keys=all_keys),
        CropForegroundd(keys=all_keys, source_key=image_keys[0]),
        Orientationd(keys=all_keys, axcodes="RA"),
        SpatialPadd(keys=all_keys, spatial_size=spatial_size),
        RandZoomd(
            keys=all_keys,
            min_zoom=0.7,
            max_zoom=1.5,
            mode=("bilinear",) * len(image_keys) + ("nearest",),
            align_corners=(True,) * len(image_keys) + (None,),
            prob=0.3,
        ),
        # RandRotated(
        #     keys=all_keys,
        #     range_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
        #     range_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
        #     mode=("bilinear",) * len(image_keys) + ("nearest",),
        #     align_corners=(True,) * len(image_keys) + (None,),
        #     padding_mode=("border", ) * len(all_keys),
        #     prob=0.3,
        # ),
        # RandAdjustContrastd(  # same as Gamma in nnU-Net
        #     keys=image_keys,
        #     gamma=(0.7, 1.5),
        #     prob=0.3,
        # ),
        RandFlipd(all_keys, spatial_axis=[0], prob=0.5),  # Only right-left flip
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        CastToTyped(keys=all_keys, dtype=(np.float32,) * len(image_keys) + (np.uint8,)),
        ToTensord(keys=all_keys),
    ])
    return train_transform


def brats_validation_transform(image_keys, all_keys, spatial_size=(240, 240)):
    
    val_transform = Compose([
        AddChanneld(keys=all_keys),
        # CropForegroundd(keys=all_keys, source_key=image_keys[0]),
        Orientationd(keys=all_keys, axcodes="RA"),
        SpatialPadd(keys=all_keys, spatial_size=spatial_size),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        CastToTyped(keys=all_keys, dtype=(np.float32,) * len(image_keys) + (np.uint8,)),
        ToTensord(keys=all_keys),
    ])
    return val_transform

