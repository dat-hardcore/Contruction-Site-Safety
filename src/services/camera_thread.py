"""
Luồng camera và video file: dùng chung pipeline tối ưu cho cả hai nhánh.
- Detection mỗi N frame (DETECT_EVERY_N_FRAMES), YOLO dùng imgsz nhỏ (DETECT_IMGSZ) trong detector.
- Chỉ track Person (TRACK_PERSON_ONLY) giữa các lần detect → ít lag, vẫn theo dõi người.
"""
import threading
import time
from typing import Callable, List, Optional, Union

import cv2
import numpy as np

from src.config.settings import CAMERA_INDEX, DETECT_EVERY_N_FRAMES, TRACK_PERSON_ONLY
from src.models import DetectionResult, Violation
from src.services.detector import PPEDetector, draw_tracked_boxes

# Dataset: Person = index 3
PERSON_CLS_ID = 3


class CameraThread:
    """
    Chạy camera hoặc video file trong thread riêng, gọi callback khi có frame mới.
    - source: int = camera index (0 = webcam), str = đường dẫn file video (.mp4, .avi, ...).
    - on_video_end: gọi khi đọc hết file video (không dùng khi là camera).
    """

    def __init__(
        self,
        on_frame: Callable[[DetectionResult], None],
        source: Union[int, str] = CAMERA_INDEX,
        detector: Optional[PPEDetector] = None,
        on_video_end: Optional[Callable[[], None]] = None,
    ):
        self.on_frame = on_frame
        self.source = source  # int = camera index, str = video path
        self.detector = detector or PPEDetector()
        self.on_video_end = on_video_end
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._last_tracked_boxes: List[dict] = []
        self._last_violations: List[Violation] = []
        self._multi_tracker = None  # cv2.legacy.MultiTracker, chỉ chứa box Person
        self._person_indices: List[int] = []  # chỉ số trong _last_tracked_boxes là Person

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        if isinstance(self.source, str):
            self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(int(self.source))
        if not self._cap.isOpened():
            self.on_frame(DetectionResult(frame=np.zeros((480, 640, 3), dtype=np.uint8), violations=[], fps=0))
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def _run(self):
        # Cùng pipeline tối ưu cho cả camera (source=int) và video file (source=str)
        fps_start = time.perf_counter()
        fps_frames = 0
        fps_value = 0.0
        is_video_file = isinstance(self.source, str)
        video_fps = self._cap.get(cv2.CAP_PROP_FPS) if is_video_file else 30
        frame_delay = 1.0 / video_fps if video_fps > 0 else 0.033
        detect_every = max(1, int(DETECT_EVERY_N_FRAMES))
        self._frame_index = 0
        self._last_tracked_boxes = []
        self._last_violations = []
        self._multi_tracker = None
        self._person_indices = []

        while not self._stop.is_set() and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                if is_video_file and callable(self.on_video_end):
                    try:
                        self.on_video_end()
                    except Exception:
                        pass
                break
            fps_frames += 1
            if fps_frames >= 10:
                fps_value = fps_frames / (time.perf_counter() - fps_start)
                fps_frames = 0
                fps_start = time.perf_counter()

            do_detect = self._frame_index % detect_every == 0 or not self._last_tracked_boxes
            if do_detect:
                frame_with_boxes, violations, tracked_boxes = self.detector.detect_and_draw_all(frame)
                self._last_violations = violations
                self._last_tracked_boxes = tracked_boxes
                if TRACK_PERSON_ONLY and tracked_boxes:
                    self._person_indices = [i for i, d in enumerate(tracked_boxes) if d.get("cls_id") == PERSON_CLS_ID]
                    self._multi_tracker = self._create_person_tracker(frame, tracked_boxes)
                else:
                    self._person_indices = []
                    self._multi_tracker = None
            else:
                if self._last_tracked_boxes and self._multi_tracker is not None and self._person_indices:
                    ok, new_rects = self._multi_tracker.update(frame)
                    if ok and len(new_rects) == len(self._person_indices):
                        for j, (x, y, w, h) in enumerate(new_rects):
                            if j < len(self._person_indices):
                                idx = self._person_indices[j]
                                x1 = max(0, int(x))
                                y1 = max(0, int(y))
                                x2 = min(frame.shape[1], int(x + w))
                                y2 = min(frame.shape[0], int(y + h))
                                self._last_tracked_boxes[idx] = {
                                    **self._last_tracked_boxes[idx],
                                    "bbox": (x1, y1, x2, y2),
                                }
                    frame_with_boxes = draw_tracked_boxes(frame, self._last_tracked_boxes, self._last_violations)
                elif self._last_tracked_boxes:
                    frame_with_boxes = draw_tracked_boxes(frame, self._last_tracked_boxes, self._last_violations)
                else:
                    frame_with_boxes = frame
                    self._last_violations = []

            self._frame_index += 1
            result = DetectionResult(
                frame=frame_with_boxes,
                violations=self._last_violations,
                fps=fps_value,
            )
            try:
                self.on_frame(result)
            except Exception:
                pass
            time.sleep(frame_delay if is_video_file else 0.03)
        if self._cap:
            self._cap.release()
            self._cap = None

    def _create_person_tracker(self, frame: np.ndarray, tracked_boxes: List[dict]):
        """Chỉ track box Person (cls_id=3) → ít tracker, ít lag, vẫn theo dõi người."""
        person_boxes = [tracked_boxes[i] for i in self._person_indices if i < len(tracked_boxes)]
        if not person_boxes:
            return None
        try:
            legacy = getattr(cv2, "legacy", None)
            if legacy is None:
                return None
            multi = legacy.MultiTracker_create()
            for d in person_boxes:
                bbox = d.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                roi = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                tracker = legacy.TrackerKCF_create()
                multi.add(tracker, frame, roi)
            return multi
        except Exception:
            return None
