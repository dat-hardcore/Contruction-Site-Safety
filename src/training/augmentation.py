"""
Data Augmentation cho YOLOv26 — mô phỏng nhiều điều kiện môi trường (ánh sáng, thời tiết, nhiễu, hình học, che khuất).
Box format: numpy array (N, 5) — [class_id, x_center, y_center, width, height] normalized trong [0, 1].
Nếu chỉ có (N, 4) thì coi là [x_center, y_center, width, height].
"""
from __future__ import annotations

import abc
import random
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


def _clip_boxes_xyxy(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Clip boxes (N,4) xyxy to [0,1]."""
    out = np.clip(boxes_xyxy, 0.0, 1.0)
    return out


def _xywh_to_xyxy(boxes: np.ndarray, has_class: bool) -> np.ndarray:
    """(N,5) or (N,4) normalized xywh -> (N,4) xyxy."""
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if has_class:
        xc, yc, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    else:
        xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    """(N,4) xyxy -> (N,4) xywh normalized."""
    if boxes_xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([xc, yc, w, h], axis=1)


def _filter_boxes(boxes_xyxy: np.ndarray, min_area: float = 1e-6) -> np.ndarray:
    """Loại box có diện tích quá nhỏ."""
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    return boxes_xyxy[areas >= min_area]


class BaseAugment(abc.ABC):
    """Base class cho mọi augment; có thể thay đổi cả ảnh và box."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        has_class: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        image: BGR (H, W, 3); boxes: (N, 5) [cls, xc, yc, w, h] hoặc (N, 4) [xc, yc, w, h].
        Returns (image, boxes) cùng format.
        """
        if random.random() >= self.p:
            return image, boxes.copy()
        return self.apply(image, boxes, has_class)

    @abc.abstractmethod
    def apply(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        has_class: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


# ---------------------------------------------------------------------------
# Ánh sáng & màu (không đổi box)
# ---------------------------------------------------------------------------


class RandomBrightness(BaseAugment):
    """Điều kiện sáng/tối (trời nắng, trong bóng râm, đêm)."""

    def __init__(self, delta: float = 0.25, p: float = 0.5):
        super().__init__(p)
        self.delta = delta

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        delta = random.uniform(-self.delta, self.delta)
        out = cv2.convertScaleAbs(image, alpha=1.0, beta=255 * delta)
        return np.clip(out, 0, 255).astype(np.uint8), boxes.copy()


class RandomContrast(BaseAugment):
    """Tương phản thay đổi (màn hình, điều kiện ánh sáng khác nhau)."""

    def __init__(self, range_contrast: Tuple[float, float] = (0.75, 1.35), p: float = 0.5):
        super().__init__(p)
        self.range_contrast = range_contrast

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        alpha = random.uniform(*self.range_contrast)
        out = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return np.clip(out, 0, 255).astype(np.uint8), boxes.copy()


class RandomGamma(BaseAugment):
    """Mô phỏng low-light / overexposed (gamma)."""

    def __init__(self, range_gamma: Tuple[float, float] = (0.5, 2.0), p: float = 0.5):
        super().__init__(p)
        self.range_gamma = range_gamma

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        gamma = random.uniform(*self.range_gamma)
        inv = 1.0 / max(gamma, 1e-3)
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(image, table)
        return out, boxes.copy()


class RandomHSV(BaseAugment):
    """Thay đổi Hue / Saturation / Value — nắng, đèn vàng, bụi bẩn màu."""

    def __init__(
        self,
        h: float = 0.03,
        s: float = 0.4,
        v: float = 0.35,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.h, self.s, self.v = h, s, v

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        dh = random.uniform(-self.h, self.h)
        ds = random.uniform(1 - self.s, 1 + self.s)
        dv = random.uniform(1 - self.v, 1 + self.v)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + dh * 180) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * ds, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * dv, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out, boxes.copy()


# ---------------------------------------------------------------------------
# Nhiễu & mờ (môi trường khói bụi, mưa, kính bẩn)
# ---------------------------------------------------------------------------


class GaussianNoise(BaseAugment):
    """Nhiễu sensor / điều kiện ánh sáng kém."""

    def __init__(self, var_limit: Tuple[float, float] = (5, 25), p: float = 0.4):
        super().__init__(p)
        self.var_limit = var_limit

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        var = random.uniform(*self.var_limit)
        noise = np.random.randn(*image.shape) * np.sqrt(var)
        out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out, boxes.copy()


class GaussianBlur(BaseAugment):
    """Mờ nhẹ — sương, kính bẩn, out-of-focus."""

    def __init__(self, kernel_range: Tuple[int, int] = (3, 7), p: float = 0.35):
        super().__init__(p)
        self.kernel_range = kernel_range

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        k = random.randrange(self.kernel_range[0], self.kernel_range[1] + 1) | 1
        out = cv2.GaussianBlur(image, (k, k), 0)
        return out, boxes.copy()


class MotionBlur(BaseAugment):
    """Mờ chuyển động — camera hoặc người di chuyển nhanh."""

    def __init__(self, max_kernel: int = 9, p: float = 0.3):
        super().__init__(p)
        self.max_kernel = max_kernel

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        k = random.randrange(3, min(self.max_kernel, min(image.shape[:2])), 2)
        kernel = np.zeros((k, k), dtype=np.float32)
        if random.random() > 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        out = cv2.filter2D(image, -1, kernel)
        return out, boxes.copy()


# ---------------------------------------------------------------------------
# Hình học (cần transform box)
# ---------------------------------------------------------------------------


class HorizontalFlip(BaseAugment):
    """Lật ngang — góc quay camera khác nhau."""

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        out = cv2.flip(image, 1)
        xyxy = _xywh_to_xyxy(boxes, has_class)
        xyxy[:, [0, 2]] = 1.0 - xyxy[:, [2, 0]]
        xywh = _xyxy_to_xywh(_clip_boxes_xyxy(xyxy))
        if has_class:
            new_boxes = np.hstack([boxes[:, :1], xywh])
        else:
            new_boxes = xywh
        return out, new_boxes.astype(np.float32)


class VerticalFlip(BaseAugment):
    """Lật dọc — ít dùng nhưng giúp đa dạng."""

    def __init__(self, p: float = 0.35):
        super().__init__(p)

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        out = cv2.flip(image, 0)
        xyxy = _xywh_to_xyxy(boxes, has_class)
        xyxy[:, [1, 3]] = 1.0 - xyxy[:, [3, 1]]
        xywh = _xyxy_to_xywh(_clip_boxes_xyxy(xyxy))
        if has_class:
            new_boxes = np.hstack([boxes[:, :1], xywh])
        else:
            new_boxes = xywh
        return out, new_boxes.astype(np.float32)


class RandomRotate(BaseAugment):
    """Xoay nhẹ — camera nghiêng, góc chụp khác nhau."""

    def __init__(self, angle_range: Tuple[float, float] = (-12, 12), p: float = 0.4):
        super().__init__(p)
        self.angle_range = angle_range

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        angle = random.uniform(*self.angle_range)
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        out = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        xyxy = _xywh_to_xyxy(boxes, has_class)
        xyxy_px = xyxy * np.array([w, h, w, h])
        ones = np.ones((len(xyxy_px), 1))
        pts = xyxy_px[:, :2]
        pts_aug = (M[:, :2] @ pts.T).T + M[:, 2]
        pts2_aug = (M[:, :2] @ xyxy_px[:, 2:4].T).T + M[:, 2]
        xyxy_new = np.hstack([pts_aug, pts2_aug]) / np.array([w, h, w, h])
        xyxy_new = _clip_boxes_xyxy(xyxy_new)
        xywh = _xyxy_to_xywh(xyxy_new)
        if has_class:
            new_boxes = np.hstack([boxes[:, :1], xywh])
        else:
            new_boxes = xywh
        return out, new_boxes.astype(np.float32)


class RandomScaleCrop(BaseAugment):
    """Scale + crop — khoảng cách camera xa/gần, crop ngẫu nhiên."""

    def __init__(self, scale_range: Tuple[float, float] = (0.85, 1.15), p: float = 0.45):
        super().__init__(p)
        self.scale_range = scale_range

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        scale = random.uniform(*self.scale_range)
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < 10 or new_h < 10:
            return image, boxes.copy()
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if scale > 1:
            x0 = random.randint(0, new_w - w)
            y0 = random.randint(0, new_h - h)
            out = resized[y0 : y0 + h, x0 : x0 + w]
            xyxy = _xywh_to_xyxy(boxes, has_class)
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * scale - x0 / w
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * scale - y0 / h
        else:
            pad_w, pad_h = w - new_w, h - new_h
            left = random.randint(0, max(0, pad_w))
            top = random.randint(0, max(0, pad_h))
            out = np.zeros_like(image)
            out[top : top + new_h, left : left + new_w] = resized
            xyxy = _xywh_to_xyxy(boxes, has_class)
            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left / w) * (w / new_w)
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top / h) * (h / new_h)
        xyxy = _clip_boxes_xyxy(xyxy)
        xywh = _xyxy_to_xywh(xyxy)
        if has_class:
            new_boxes = np.hstack([boxes[:, :1], xywh]).astype(np.float32)
        else:
            new_boxes = xywh.astype(np.float32)
        return out, new_boxes


class RandomPerspective(BaseAugment):
    """Biến dạng perspective nhẹ — góc chụp xiên, gần với thực tế công trường."""

    def __init__(self, distort_scale: float = 0.08, p: float = 0.35):
        super().__init__(p)
        self.distort_scale = distort_scale

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        d = self.distort_scale * min(w, h)
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.array(src + np.random.uniform(-d, d, (4, 2)), dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        xyxy = _xywh_to_xyxy(boxes, has_class)
        xyxy_px = xyxy * np.array([w, h, w, h])
        pts = xyxy_px.reshape(-1, 2)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_new = (M @ pts_h.T).T
        pts_new = pts_new[:, :2] / np.clip(pts_new[:, 2:3], 1e-6, None)
        xyxy_new = pts_new.reshape(-1, 4) / np.array([w, h, w, h])
        xyxy_new = _clip_boxes_xyxy(xyxy_new)
        xywh = _xyxy_to_xywh(xyxy_new)
        if has_class:
            new_boxes = np.hstack([boxes[:, :1], xywh])
        else:
            new_boxes = xywh
        return out, new_boxes.astype(np.float32)


# ---------------------------------------------------------------------------
# Che khuất (occlusion)
# ---------------------------------------------------------------------------


class RandomCutout(BaseAugment):
    """Xóa ngẫu nhiên vùng hình chữ nhật — vật che, bụi, lỗi sensor."""

    def __init__(self, n_holes: Tuple[int, int] = (1, 4), size_ratio: Tuple[float, float] = (0.02, 0.12), p: float = 0.4):
        super().__init__(p)
        self.n_holes = n_holes
        self.size_ratio = size_ratio

    def apply(self, image: np.ndarray, boxes: np.ndarray, has_class: bool) -> Tuple[np.ndarray, np.ndarray]:
        out = image.copy()
        h, w = image.shape[:2]
        n = random.randint(*self.n_holes)
        for _ in range(n):
            r = random.uniform(*self.size_ratio)
            sh, sw = int(h * r), int(w * r)
            if sh < 2 or sw < 2:
                continue
            x1 = random.randint(0, max(0, w - sw))
            y1 = random.randint(0, max(0, h - sh))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            out[y1 : y1 + sh, x1 : x1 + sw] = color
        return out, boxes.copy()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AugmentPipeline:
    """
    Pipeline augmentation: áp dụng lần lượt các transform (mỗi cái có xác suất p).
    Giúp mô hình thích nghi: ánh sáng thay đổi, thời tiết (mờ, nhiễu), góc máy (flip, xoay, perspective), che khuất.
    """

    def __init__(self, transforms: List[BaseAugment], seed: Optional[int] = None):
        self.transforms = transforms
        self.seed = seed

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        has_class: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        image: BGR (H,W,3).
        boxes: (N,5) [class_id, x_center, y_center, width, height] hoặc (N,4) [xc,yc,w,h], normalized [0,1].
        """
        if self.seed is not None:
            random.seed(self.seed)
        out_img = image
        out_boxes = boxes.copy()
        for t in self.transforms:
            out_img, out_boxes = t(out_img, out_boxes, has_class)
        return out_img, out_boxes

    @classmethod
    def default_training(cls, seed: Optional[int] = None) -> "AugmentPipeline":
        """Pipeline mặc định cho training PPE / công trường: đủ ánh sáng, thời tiết, hình học, occlusion."""
        return cls(
            transforms=[
                RandomBrightness(delta=0.25, p=0.6),
                RandomContrast(range_contrast=(0.8, 1.25), p=0.5),
                RandomGamma(range_gamma=(0.6, 1.8), p=0.4),
                RandomHSV(h=0.02, s=0.35, v=0.35, p=0.5),
                GaussianNoise(var_limit=(5, 20), p=0.35),
                GaussianBlur(kernel_range=(3, 6), p=0.3),
                MotionBlur(max_kernel=7, p=0.25),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.25),
                RandomRotate(angle_range=(-10, 10), p=0.4),
                RandomScaleCrop(scale_range=(0.9, 1.12), p=0.4),
                RandomPerspective(distort_scale=0.06, p=0.3),
                RandomCutout(n_holes=(1, 3), size_ratio=(0.02, 0.1), p=0.35),
            ],
            seed=seed,
        )


def boxes_yolo_lines_to_array(lines: List[str]) -> Tuple[np.ndarray, bool]:
    """
    Đọc từ file label YOLO (mỗi dòng: class x_center y_center width height).
    Trả về (boxes (N,5), has_class=True).
    """
    boxes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append([cls_id, xc, yc, w, h])
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32), True
    return np.array(boxes, dtype=np.float32), True


def boxes_array_to_yolo_lines(boxes: np.ndarray, has_class: bool = True) -> List[str]:
    """Chuyển boxes (N,5) hoặc (N,4) thành danh sách dòng text cho file label YOLO."""
    lines = []
    for i in range(len(boxes)):
        if has_class and boxes.shape[1] >= 5:
            cls_id = int(boxes[i, 0])
            xc, yc, w, h = boxes[i, 1], boxes[i, 2], boxes[i, 3], boxes[i, 4]
        else:
            cls_id = 0
            xc, yc, w, h = boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines
