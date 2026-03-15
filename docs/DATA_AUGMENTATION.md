# Data Augmentation cho YOLOv26 — Thích nghi điều kiện môi trường

Module augmentation giúp **làm khó** mô hình khi train, mô phỏng nhiều điều kiện thực tế (ánh sáng, thời tiết, nhiễu, góc máy, che khuất) để model PPE (mũ/áo bảo hộ) hoạt động ổn định trên công trường.

## Các nhóm augmentation

| Nhóm | Transform | Mục đích |
|------|-----------|----------|
| **Ánh sáng** | RandomBrightness, RandomContrast, RandomGamma | Nắng gắt, bóng râm, low-light, đèn vàng |
| **Màu** | RandomHSV (hue, saturation, value) | Thay đổi ánh đèn, bụi bẩn màu |
| **Nhiễu / mờ** | GaussianNoise, GaussianBlur, MotionBlur | Sensor noise, sương, kính bẩn, chuyển động |
| **Hình học** | HorizontalFlip, VerticalFlip, RandomRotate, RandomScaleCrop, RandomPerspective | Góc quay camera, xa/gần, xiên |
| **Che khuất** | RandomCutout | Vật che, bụi, lỗi vùng ảnh |

Box luôn được transform đúng khi cần (flip, rotate, scale/crop, perspective); format **YOLO chuẩn**: `[class_id, x_center, y_center, width, height]` normalized trong [0, 1].

## Cách dùng trong code

```python
from src.training.augmentation import AugmentPipeline
import cv2

# Pipeline mặc định (đã cân chỉnh cho PPE / công trường)
pipeline = AugmentPipeline.default_training(seed=42)

image = cv2.imread("path/to/image.jpg")  # BGR
# boxes: (N, 5) — [class_id, xc, yc, w, h] normalized
boxes = np.array([
    [0, 0.5, 0.5, 0.2, 0.3],   # class 0, 1 box
])

aug_image, aug_boxes = pipeline(image, boxes, has_class=True)
```

## Chuẩn bị dataset đã augment (offline)

Tạo thêm ảnh + label đã augment từ dataset YOLO có sẵn:

```bash
# Từ thư mục gốc dự án
python -m src.training.prepare_augmented_dataset \
  --images data/train/images \
  --labels data/train/labels \
  --out data/train_aug \
  --num 2 \
  --copy-original
```

- `--num 2`: mỗi ảnh sinh thêm 2 bản augment (tổng 2 ảnh mới/ảnh; nếu có `--copy-original` thì thêm cả ảnh gốc).
- `--copy-original`: copy cả ảnh và label gốc vào `out/images` và `out/labels`.
- Kết quả: `data/train_aug/images/`, `data/train_aug/labels/` — dùng làm thư mục train (hoặc trộn với gốc).

Sau đó trong `data.yaml` hoặc lệnh train Ultralytics trỏ `path` tới thư mục chứa `train_aug` (hoặc train gốc + train_aug nếu bạn gộp).

## Tích hợp với Ultralytics khi train

1. **Dùng dataset đã augment (offline)**  
   Chạy script trên rồi trỏ `data.yaml` tới thư mục đã augment.

2. **Augmentation sẵn trong Ultralytics**  
   Ultralytics đã có HSV, flip, mosaic, v.v. Bạn có thể tăng cường thêm bằng cách:
   - Tạo nhiều ảnh đa dạng hơn bằng module này (offline), hoặc
   - Truyền **custom Albumentations** khi gọi `model.train(..., augmentations=[...])` (xem tài liệu Ultralytics).

Module này bổ sung các điều kiện **môi trường** (ánh sáng, nhiễu, mờ, perspective, cutout) rõ ràng và có thể chỉnh từng tham số (p, cường độ) trong `AugmentPipeline.default_training()` hoặc tự tạo pipeline với `AugmentPipeline(transforms=[...])`.
