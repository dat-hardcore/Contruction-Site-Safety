"""
Data Augmentation cho training YOLO — làm khó mô hình để thích nghi nhiều điều kiện môi trường.
Dùng khi chuẩn bị dataset hoặc tích hợp vào pipeline train Ultralytics.
Box format: normalized (x_center, y_center, width, height) trong [0, 1]; mỗi dòng có thể kèm class_id.
"""

from src.training.augmentation import (
    AugmentPipeline,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    RandomHSV,
    GaussianNoise,
    GaussianBlur,
    MotionBlur,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate,
    RandomScaleCrop,
    RandomPerspective,
    RandomCutout,
)

__all__ = [
    "AugmentPipeline",
    "RandomBrightness",
    "RandomContrast",
    "RandomGamma",
    "RandomHSV",
    "GaussianNoise",
    "GaussianBlur",
    "MotionBlur",
    "HorizontalFlip",
    "VerticalFlip",
    "RandomRotate",
    "RandomScaleCrop",
    "RandomPerspective",
    "RandomCutout",
]
