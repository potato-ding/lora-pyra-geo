"""Teacher-specific image transforms."""

import inspect

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def _transform_supports(transform_cls, arg_name):
    try:
        return arg_name in inspect.signature(transform_cls).parameters
    except (TypeError, ValueError):
        return False


def _image_compression_90_100(p=0.5):
    if _transform_supports(A.ImageCompression, "quality_range"):
        return A.ImageCompression(quality_range=(90, 100), p=p)
    return A.ImageCompression(quality_lower=90, quality_upper=100, p=p)


def _paired_cross_view_coarse_dropout(img_size, p=1.0):
    min_h = int(0.1 * img_size[0])
    max_h = int(0.2 * img_size[0])
    min_w = int(0.1 * img_size[1])
    max_w = int(0.2 * img_size[1])
    if _transform_supports(A.CoarseDropout, "num_holes_range"):
        return A.CoarseDropout(
            num_holes_range=(10, 25),
            hole_height_range=(min_h, max_h),
            hole_width_range=(min_w, max_w),
            p=p,
        )
    return A.CoarseDropout(
        max_holes=25,
        max_height=max_h,
        max_width=max_w,
        min_holes=10,
        min_height=min_h,
        min_width=min_w,
        p=p,
    )


def get_paired_cross_view_train_transforms(
    img_size=[224, 224],
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    def common_transforms(grid_ratio):
        return [
            _image_compression_90_100(p=0.5),
            A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.15,
                p=0.5,
            ),
            A.OneOf([
                A.AdvancedBlur(p=1.0),
                A.Sharpen(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.GridDropout(ratio=grid_ratio, p=1.0),
                _paired_cross_view_coarse_dropout(img_size, p=1.0),
            ], p=0.3),
        ]

    train_sat_transforms = A.Compose([
        *common_transforms(grid_ratio=0.4),
        A.RandomRotate90(p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    train_drone_transforms = A.Compose([
        *common_transforms(grid_ratio=0.4),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return train_sat_transforms, train_drone_transforms


def get_paired_cross_view_val_transforms(
    img_size=[224, 224],
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    return A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


__all__ = [
    "get_paired_cross_view_train_transforms",
    "get_paired_cross_view_val_transforms",
]
