"""
Chuẩn bị dataset đã augment cho YOLO — đọc thư mục ảnh + label (format YOLO), sinh thêm ảnh đã augment và ghi ra thư mục đích.
Cách chạy (từ thư mục gốc dự án):
  python -m src.training.prepare_augmented_dataset --images data/train/images --labels data/train/labels --out data/train_aug --num 2
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

# Thêm gốc dự án vào path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.training.augmentation import (
    AugmentPipeline,
    boxes_array_to_yolo_lines,
    boxes_yolo_lines_to_array,
)


def _find_pairs(images_dir: str, labels_dir: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")):
    """Trả về list (path_ảnh, path_label) với base name trùng nhau."""
    pairs = []
    for name in os.listdir(images_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() not in extensions:
            continue
        img_path = os.path.join(images_dir, name)
        label_path = os.path.join(labels_dir, base + ".txt")
        if not os.path.isfile(img_path) or not os.path.isfile(label_path):
            continue
        pairs.append((img_path, label_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Tạo dataset đã augment cho YOLO (PPE / công trường).")
    parser.add_argument("--images", required=True, help="Thư mục chứa ảnh (YOLO dataset/images)")
    parser.add_argument("--labels", required=True, help="Thư mục chứa label .txt (YOLO dataset/labels)")
    parser.add_argument("--out", required=True, help="Thư mục đích: out/images và out/labels")
    parser.add_argument("--num", type=int, default=2, help="Số bản augment thêm cho mỗi ảnh (mặc định 2)")
    parser.add_argument("--copy-original", action="store_true", help="Copy cả ảnh gốc vào out (mặc định chỉ ghi ảnh augment)")
    parser.add_argument("--ext", default=".jpg", help="Đuôi ảnh đích (mặc định .jpg)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (tùy chọn)")
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images)
    labels_dir = os.path.abspath(args.labels)
    out_dir = os.path.abspath(args.out)
    out_images = os.path.join(out_dir, "images")
    out_labels = os.path.join(out_dir, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    pairs = _find_pairs(images_dir, labels_dir)
    if not pairs:
        print("Không tìm thấy cặp (ảnh, label) nào. Kiểm tra --images và --labels.")
        return 1

    pipeline = AugmentPipeline.default_training(seed=args.seed)
    total_saved = 0

    for img_path, label_path in pairs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        image = cv2.imread(img_path)
        if image is None:
            print("Bỏ qua (không đọc được ảnh):", img_path)
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        boxes, has_class = boxes_yolo_lines_to_array(lines)
        if args.copy_original:
            out_img_path = os.path.join(out_images, base + args.ext)
            out_label_path = os.path.join(out_labels, base + ".txt")
            cv2.imwrite(out_img_path, image)
            with open(out_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(boxes_array_to_yolo_lines(boxes, has_class)))
            total_saved += 1
        for k in range(args.num):
            aug_img, aug_boxes = pipeline(image, boxes, has_class)
            out_base = f"{base}_aug{k}"
            out_img_path = os.path.join(out_images, out_base + args.ext)
            out_label_path = os.path.join(out_labels, out_base + ".txt")
            cv2.imwrite(out_img_path, aug_img)
            with open(out_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(boxes_array_to_yolo_lines(aug_boxes, has_class)))
            total_saved += 1
    print(f"Đã ghi {total_saved} ảnh + label vào {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
