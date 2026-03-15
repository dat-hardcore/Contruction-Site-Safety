"""
Nhận diện vi phạm trang bị bảo hộ trên ảnh từ camera.
Dataset nhãn: "Gloves", "Helmet", "Non-Helmet", "Person", "Shoes", "Vest", "bare-arms".
Vẽ toàn bộ bounding box model detect (để kiểm tra mô hình); vi phạm (Non-Helmet, bare-arms) vẫn dùng cho KPI/danh sách.
"""
import uuid
from datetime import datetime
from typing import List, Optional, Tuple, Any

import cv2
import numpy as np

from src.config.settings import CONFIDENCE_THRESHOLD, DETECT_IMGSZ, MODEL_PATH, YOLO26_MODEL
from src.models import Violation, ViolationType


# Thứ tự class phải khớp với file train (data.yaml) khi train model
DATASET_CLASS_NAMES = [
    "Gloves",      # 0
    "Helmet",      # 1
    "Non-Helmet",  # 2  → vi phạm
    "Person",      # 3
    "Shoes",       # 4
    "Vest",        # 5
    "bare-arms",   # 6  → vi phạm
]

# Map class index → loại vi phạm (chỉ class thể hiện vi phạm trực tiếp)
CLASS_INDEX_TO_VIOLATION = {
    2: ViolationType.MISSING_HELMET,   # Non-Helmet
    6: ViolationType.MISSING_VEST,     # bare-arms (thiếu áo bảo hộ / để lộ tay)
}


class PPEDetector:
    """
    Phát hiện vi phạm PPE theo dataset: Non-Helmet, bare-arms.
    - MODEL_PATH = file .pt (model train với 7 class trên) → dùng model đó.
    - MODEL_PATH = None → dùng YOLO26 pretrained (không có class PPE, chạy chế độ mẫu khi cần).
    """

    CLASS_NAMES = DATASET_CLASS_NAMES
    CLASS_TO_VIOLATION = CLASS_INDEX_TO_VIOLATION

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        yolo26_model: Optional[str] = None,
    ):
        self.model_path = model_path if model_path is not None else MODEL_PATH
        self.yolo26_model = yolo26_model or YOLO26_MODEL
        self.conf_threshold = confidence_threshold
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if self.model_path:
                self._model = YOLO(self.model_path)
                print(f"[PPEDetector] Đã load model PPE: {self.model_path}")
                print(f"[PPEDetector] Nhãn: {self.CLASS_NAMES}. Vi phạm: Non-Helmet (2), bare-arms (6).")
            else:
                self._model = YOLO(self.yolo26_model)
                print(f"[PPEDetector] YOLOv26: {self.yolo26_model}. Đặt MODEL_PATH để dùng model PPE.")
        except Exception as e:
            print(f"[PPEDetector] Lỗi load model: {e}. Chạy chế độ mẫu.")
            self._model = None

    def detect(self, frame: np.ndarray) -> List[Violation]:
        """
        Nhận diện vi phạm trên frame BGR.
        Chỉ báo vi phạm khi model detect class Non-Helmet (2) hoặc bare-arms (6).
        """
        if self._model is not None:
            return self._detect_yolo(frame)
        return self._detect_demo(frame)

    def detect_and_draw_all(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Violation], List[dict]]:
        """
        Chạy model, vẽ TOÀN BỘ bounding box (mọi class) lên frame.
        Trả về (frame đã vẽ, danh sách vi phạm, danh sách box để tracking).
        """
        if self._model is not None:
            return self._detect_yolo_and_draw_all(frame)
        violations = self._detect_demo(frame)
        return draw_violations_on_frame(frame, violations), violations, []

    def _detect_yolo_and_draw_all(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Violation], List[dict]]:
        results = self._model.predict(
            frame, conf=self.conf_threshold, verbose=False, stream=False, imgsz=DETECT_IMGSZ
        )
        violations = []
        tracked_boxes: List[dict] = []
        out = frame.copy()
        class_names = getattr(self._model, "names", None) or DATASET_CLASS_NAMES
        if isinstance(class_names, dict):
            max_id = max(class_names.keys()) if class_names else 0
            names_list = [class_names.get(i, str(i)) for i in range(max_id + 1)]
        else:
            names_list = list(class_names) if class_names else []
        colors = [
            (255, 128, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 128),
        ]
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = names_list[cls_id] if cls_id < len(names_list) else str(cls_id)
                color = colors[cls_id % len(colors)]
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {int(conf*100)}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(out, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                tracked_boxes.append({"bbox": (x1, y1, x2, y2), "name": name, "conf": conf, "cls_id": cls_id})
                if cls_id in self.CLASS_TO_VIOLATION:
                    violations.append(
                        Violation(
                            id=str(uuid.uuid4()),
                            violation_type=self.CLASS_TO_VIOLATION[cls_id],
                            confidence=conf,
                            timestamp=datetime.now(),
                            bbox=(x1, y1, x2, y2),
                        )
                    )
        return out, violations, tracked_boxes

    def _detect_yolo(self, frame: np.ndarray) -> List[Violation]:
        results = self._model.predict(
            frame, conf=self.conf_threshold, verbose=False, stream=False, imgsz=DETECT_IMGSZ
        )
        violations = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.CLASS_TO_VIOLATION:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                violations.append(
                    Violation(
                        id=str(uuid.uuid4()),
                        violation_type=self.CLASS_TO_VIOLATION[cls_id],
                        confidence=conf,
                        timestamp=datetime.now(),
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return violations

    def _detect_demo(self, frame: np.ndarray) -> List[Violation]:
        """Chế độ mẫu khi không load được model."""
        h, w = frame.shape[:2]
        violations = []
        np.random.seed(hash(frame.tobytes()) % (2**32))
        for _ in range(np.random.randint(0, 3)):
            x1 = int(w * 0.2 + np.random.rand() * w * 0.3)
            y1 = int(h * 0.2 + np.random.rand() * h * 0.3)
            x2 = min(x1 + 80, w - 1)
            y2 = min(y1 + 120, h - 1)
            vt = ViolationType.MISSING_HELMET if np.random.rand() > 0.5 else ViolationType.MISSING_VEST
            violations.append(
                Violation(
                    id=str(uuid.uuid4()),
                    violation_type=vt,
                    confidence=0.85 + np.random.rand() * 0.1,
                    timestamp=datetime.now(),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return violations


def draw_tracked_boxes(
    frame: np.ndarray,
    tracked_boxes: List[dict],
    violations: List[Violation],
) -> np.ndarray:
    """Vẽ danh sách box (từ tracker) lên frame, không gọi model."""
    colors = [
        (255, 128, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 128),
    ]
    out = frame.copy()
    for d in tracked_boxes:
        bbox = d.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        name = d.get("name", "")
        conf = d.get("conf", 0)
        cls_id = d.get("cls_id", 0)
        color = colors[cls_id % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {int(conf*100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def draw_violations_on_frame(
    frame: np.ndarray,
    violations: List[Violation],
    font_scale: float = 0.55,
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ bounding box và nhãn vi phạm (tiếng Việt có dấu) lên frame BGR."""
    if not violations:
        return frame.copy()
    try:
        from PIL import Image, ImageDraw
        from src.utils.font_utils import get_vietnamese_font
        font_pil = get_vietnamese_font(size=14)
    except Exception:
        font_pil = None
    out = frame.copy()
    # Vẽ toàn bộ box bằng OpenCV
    for v in violations:
        if v.bbox is None:
            continue
        x1, y1, x2, y2 = v.bbox
        color = (0, 0, 255) if v.violation_type == ViolationType.MISSING_HELMET else (0, 255, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    # Vẽ chữ tiếng Việt bằng PIL (một lần cho cả frame)
    if font_pil is not None:
        img_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        for v in violations:
            if v.bbox is None:
                continue
            x1, y1, x2, y2 = v.bbox
            color = (0, 0, 255) if v.violation_type == ViolationType.MISSING_HELMET else (0, 255, 255)
            text = f"{v.label_vi} {v.confidence_pct}"
            tx, ty = x1 + 2, max(0, y1 - 22)
            draw.rectangle([tx - 2, ty - 2, tx + 240, ty + 18], fill=(color[2], color[1], color[0]))
            draw.text((tx, ty), text, font=font_pil, fill=(255, 255, 255))
        out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        for v in violations:
            if v.bbox is None:
                continue
            x1, y1, x2, y2 = v.bbox
            color = (0, 0, 255) if v.violation_type == ViolationType.MISSING_HELMET else (0, 255, 255)
            text = f"{v.label_vi} {v.confidence_pct}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return out
