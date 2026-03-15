"""
Convert Pascal VOC dataset to YOLO format.

Output structure:
  <output>/
    train/images, train/labels
    val/images, val/labels
    test/images, test/labels   (optional if test split exists)
    data.yaml

Example:
  python -m src.training.convert_voc_to_yolo \
    --voc-root data/VOC2028/VOC2028 \
    --output data/yolo_voc2028 \
    --classes hat,person
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC dataset to YOLO format.")
    parser.add_argument(
        "--voc-root",
        type=Path,
        required=True,
        help="VOC root folder (contains Annotations/, JPEGImages/, ImageSets/Main/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder for YOLO dataset.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split names to convert (default: train,val,test).",
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated class names to keep and class order. Empty = auto-detect from XML.",
    )
    parser.add_argument(
        "--include-difficult",
        action="store_true",
        help="Include VOC objects with difficult=1 (default: ignore difficult objects).",
    )
    parser.add_argument(
        "--keep-empty-labels",
        action="store_true",
        help="Keep images with empty labels after class filtering.",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=0,
        help="For quick test: max number of images per split (0 = no limit).",
    )
    return parser.parse_args()


def read_split_ids(path: Path) -> List[str]:
    if not path.is_file():
        return []
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item:
            ids.append(item)
    return ids


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
        p2 = images_dir / f"{stem}{ext.upper()}"
        if p2.is_file():
            return p2
    return None


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def voc_box_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, w: float, h: float) -> Sequence[float]:
    x_center = ((xmin + xmax) / 2.0) / w
    y_center = ((ymin + ymax) / 2.0) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return clamp(x_center), clamp(y_center), clamp(bw), clamp(bh)


def parse_objects(
    xml_path: Path,
    class_to_idx: dict[str, int],
    include_difficult: bool,
) -> List[str]:
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    width = float(size.findtext("width", default="0"))
    height = float(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    lines: List[str] = []
    for obj in root.findall("object"):
        if not include_difficult:
            difficult = int(obj.findtext("difficult", default="0"))
            if difficult == 1:
                continue

        class_name = (obj.findtext("name") or "").strip()
        if class_name not in class_to_idx:
            continue

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(box.findtext("xmin", default="0"))
        ymin = float(box.findtext("ymin", default="0"))
        xmax = float(box.findtext("xmax", default="0"))
        ymax = float(box.findtext("ymax", default="0"))

        if xmax <= xmin or ymax <= ymin:
            continue

        xc, yc, bw, bh = voc_box_to_yolo(xmin, ymin, xmax, ymax, width, height)
        class_id = class_to_idx[class_name]
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return lines


def detect_classes(annotation_files: Iterable[Path]) -> List[str]:
    counts: Counter[str] = Counter()
    for xml_path in annotation_files:
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            continue
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name:
                counts[name] += 1
    return [name for name, _ in counts.most_common()]


def write_data_yaml(output_dir: Path, split_names: Sequence[str], classes: Sequence[str]) -> None:
    lines = ["path: ."]
    if "train" in split_names:
        lines.append("train: train/images")
    if "val" in split_names:
        lines.append("val: val/images")
    if "test" in split_names:
        lines.append("test: test/images")
    lines.append("names:")
    for idx, name in enumerate(classes):
        lines.append(f"  {idx}: {name}")
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    voc_root = args.voc_root.resolve()
    output_dir = args.output.resolve()

    ann_dir = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"
    split_dir = voc_root / "ImageSets" / "Main"

    if not ann_dir.is_dir() or not images_dir.is_dir() or not split_dir.is_dir():
        raise FileNotFoundError(
            f"VOC folder must contain Annotations/, JPEGImages/, ImageSets/Main/. Got: {voc_root}"
        )

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not split_names:
        raise ValueError("No split names provided.")

    all_xml_files = list(ann_dir.glob("*.xml"))
    if not all_xml_files:
        raise FileNotFoundError(f"No XML files found in: {ann_dir}")

    if args.classes.strip():
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        classes = detect_classes(all_xml_files)
    if not classes:
        raise ValueError("No class names detected/provided.")

    class_to_idx = {name: i for i, name in enumerate(classes)}

    print(f"[INFO] VOC root: {voc_root}")
    print(f"[INFO] Output:   {output_dir}")
    print(f"[INFO] Splits:   {split_names}")
    print(f"[INFO] Classes:  {classes}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for split in split_names:
        ids = read_split_ids(split_dir / f"{split}.txt")
        if args.max_per_split > 0:
            ids = ids[: args.max_per_split]
        if not ids:
            print(f"[WARN] Split '{split}' is empty or missing.")
            continue

        out_img_dir = output_dir / split / "images"
        out_lbl_dir = output_dir / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        n_ok = 0
        n_missing = 0
        n_empty = 0

        for stem in ids:
            xml_path = ann_dir / f"{stem}.xml"
            img_path = find_image(images_dir, stem)

            if not xml_path.is_file() or img_path is None:
                n_missing += 1
                continue

            try:
                lines = parse_objects(
                    xml_path=xml_path,
                    class_to_idx=class_to_idx,
                    include_difficult=args.include_difficult,
                )
            except Exception:
                n_missing += 1
                continue

            if not lines and not args.keep_empty_labels:
                n_empty += 1
                continue

            shutil.copy2(img_path, out_img_dir / img_path.name)
            (out_lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_ok += 1

        summary[split] = {
            "requested": len(ids),
            "converted": n_ok,
            "missing_or_error": n_missing,
            "empty_filtered": n_empty,
        }

    write_data_yaml(output_dir=output_dir, split_names=split_names, classes=classes)

    print("\n[SUMMARY]")
    for split in split_names:
        if split not in summary:
            continue
        info = summary[split]
        print(
            f"  {split}: requested={info['requested']}, converted={info['converted']}, "
            f"missing_or_error={info['missing_or_error']}, empty_filtered={info['empty_filtered']}"
        )
    print(f"[DONE] data.yaml: {output_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
