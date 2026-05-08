import torchvision.transforms as transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=[224, 224], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # 通用的增强列表 (马赛克、模糊、颜色抖动等)
    list_transforms = [
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1, p=0.5),
        A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, # 对应之前的 shift_limit=0.1
            scale=(0.85, 1.15),                                    # 对应之前的 scale_limit=0.15 (1±0.15)
            rotate=(-0, 0),                                        # 对应之前的 rotate_limit=0
            p=0.5), # 注意这里不执行普通旋转
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        A.AdvancedBlur(blur_limit=(3, 7), p=0.5),
        A.GridDropout(ratio=0.3, p=0.5), # 打黑框增强
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(1, int(img_size[0]*0.3)), hole_width_range=(1, int(img_size[1]*0.3)), p=0.5),
        A.ImageCompression(quality_range=(80, 100), p=0.5),
    ]

    # 卫星图专属训练增强 (注意：包含 RandomRotate90)
    train_sat_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=3), # 对应 CV2 BICUBIC
        A.RandomRotate90(p=1.0),
        *list_transforms,
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # 无人机图专属训练增强 (注意：去掉了旋转)
    train_drone_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=3),
        *list_transforms,
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # 验证集预处理 (只调整尺寸和归一化)
    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return val_transforms, train_sat_transforms, train_drone_transforms

def get_test_transforms(img_size=[224,224]):
    val_transforms = A.Compose([
        
        A.Resize(img_size[0], img_size[1], interpolation=3), 
        
        # 确保这里的顺序是 [R, G, B]
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        ToTensorV2(),
    ])
    return val_transforms

# 转接头
def alb_transform_wrapper(image, transform):
    """
    专门用来解决 ImageFolder (输出 PIL) 和 Albumentations (需求 Numpy 且输出字典) 不兼容的问题
    """
    # 将 PIL 转换为 Numpy 数组 (ImageFolder 默认读出来就是 RGB 的，不用 cvtColor)
    image_np = np.array(image)
    # 送入 Albumentations 处理
    augmented = transform(image=image_np)
    # 提取并返回 Tensor
    return augmented["image"]