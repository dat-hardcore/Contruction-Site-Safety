"""
Hộp thoại Cài đặt: camera, khu vực, ngưỡng nhận diện, model.
"""
import json
import os
import sys

import customtkinter as ctk

from src.config import settings

if sys.platform == "win32":
    _FONT = ("Segoe UI", 12)
else:
    _FONT = ("DejaVu Sans", 12)


class SettingsDialog(ctk.CTkToplevel):
    """
    Cài đặt hệ thống:
    - Camera: index (0 = webcam mặc định)
    - Tên khu vực hiển thị trên giao diện
    - Ngưỡng confidence (độ nhạy nhận diện)
    - Throttle: thời gian tối thiểu giữa hai lần ghi nhận cùng loại vi phạm
    - Đường dẫn file model .pt (YOLO PPE)
    """

    def __init__(self, parent, on_saved=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.title("Cài đặt hệ thống")
        self.geometry("480x420")
        self.resizable(True, True)
        self.on_saved = on_saved  # callback khi người dùng bấm Lưu

        self._build_ui()
        self._load_current()
        self.transient(parent)
        self.grab_set()

    def _build_ui(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=24, pady=20)

        ctk.CTkLabel(
            main, text="Camera",
            font=ctk.CTkFont(family=_FONT[0], size=13, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self._camera_index = ctk.CTkEntry(main, placeholder_text="0 = webcam mặc định", width=320)
        self._camera_index.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            main, text="Tên khu vực (hiển thị trên giao diện)",
            font=ctk.CTkFont(family=_FONT[0], size=13, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self._area_name = ctk.CTkEntry(main, placeholder_text="Ví dụ: Khu vực A", width=320)
        self._area_name.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            main, text="Ngưỡng confidence (0.1–0.9, càng thấp càng nhạy)",
            font=ctk.CTkFont(family=_FONT[0], size=13, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self._confidence = ctk.CTkEntry(main, placeholder_text="Ví dụ: 0.25", width=320)
        self._confidence.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            main, text="Throttle (giây) – thời gian tối thiểu giữa 2 lần ghi cùng loại vi phạm",
            font=ctk.CTkFont(family=_FONT[0], size=13, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self._throttle = ctk.CTkEntry(main, placeholder_text="Ví dụ: 0.8", width=320)
        self._throttle.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            main, text="Đường dẫn file model .pt (YOLO PPE)",
            font=ctk.CTkFont(family=_FONT[0], size=13, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self._model_path = ctk.CTkEntry(main, placeholder_text="Ví dụ: helmet_best.pt hoặc đường dẫn đầy đủ", width=320)
        self._model_path.pack(fill="x", pady=(0, 20))

        btn_frame = ctk.CTkFrame(main, fg_color="transparent")
        btn_frame.pack(fill="x")
        ctk.CTkButton(btn_frame, text="Lưu", width=100, command=self._save).pack(side="right", padx=(8, 0))
        ctk.CTkButton(btn_frame, text="Hủy", width=100, fg_color="gray", command=self.destroy).pack(side="right")

    def _load_current(self):
        self._camera_index.insert(0, str(settings.CAMERA_INDEX))
        self._area_name.insert(0, settings.CAMERA_AREA_NAME)
        self._confidence.insert(0, str(settings.CONFIDENCE_THRESHOLD))
        self._throttle.insert(0, str(settings.VIOLATION_THROTTLE_SECONDS))
        mp = settings.MODEL_PATH or ""
        if mp and os.path.isabs(mp):
            self._model_path.insert(0, mp)
        else:
            self._model_path.insert(0, os.path.basename(mp) if mp else "")

    def _save(self):
        try:
            camera_index = int(self._camera_index.get().strip() or "0")
        except ValueError:
            camera_index = 0
        area_name = (self._area_name.get().strip() or settings.CAMERA_AREA_NAME)[:80]
        try:
            confidence = float(self._confidence.get().strip() or "0.25")
            confidence = max(0.1, min(0.95, confidence))
        except ValueError:
            confidence = settings.CONFIDENCE_THRESHOLD
        try:
            throttle = float(self._throttle.get().strip() or "0.8")
            throttle = max(0.2, min(10.0, throttle))
        except ValueError:
            throttle = settings.VIOLATION_THROTTLE_SECONDS
        model_path = (self._model_path.get().strip() or "").strip()
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(settings.PROJECT_ROOT, model_path)

        data = {
            "camera_index": camera_index,
            "camera_area_name": area_name,
            "confidence_threshold": confidence,
            "violation_throttle_seconds": throttle,
            "model_path": model_path or None,
        }
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        with open(settings.USER_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        settings.load_user_settings()
        if callable(self.on_saved):
            self.on_saved()
        self.destroy()
