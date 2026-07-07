"""Student image transforms."""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    img_size=[224, 224],
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    common_transforms = [
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1, p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(0, 0),
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=10,
            val_shift_limit=10,
            p=0.5,
        ),
        A.AdvancedBlur(blur_limit=(3, 7), p=0.5),
        A.GridDropout(ratio=0.3, p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(1, int(img_size[0] * 0.3)),
            hole_width_range=(1, int(img_size[1] * 0.3)),
            p=0.5,
        ),
        A.ImageCompression(quality_range=(80, 100), p=0.5),
    ]

    train_sat_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_CUBIC),
        A.RandomRotate90(p=1.0),
        *common_transforms,
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    train_drone_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_CUBIC),
        *common_transforms,
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    val_transforms = get_test_transforms(img_size=img_size, mean=mean, std=std)
    return val_transforms, train_sat_transforms, train_drone_transforms


def get_test_transforms(
    img_size=[224, 224],
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    return A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def alb_transform_wrapper(image, transform):
    augmented = transform(image=np.array(image))
    return augmented["image"]


__all__ = [
    "alb_transform_wrapper",
    "get_test_transforms",
    "get_train_transforms",
]
