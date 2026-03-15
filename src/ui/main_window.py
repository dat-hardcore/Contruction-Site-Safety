"""
Cửa sổ chính - Dashboard giám sát an toàn lao động.
Giao diện: header, KPI cards, cảnh báo, camera trực tiếp, danh sách vi phạm.
"""
from collections import deque
from datetime import datetime
from typing import Deque, List

import cv2
import customtkinter as ctk
from PIL import Image
import numpy as np

from src.config.settings import CAMERA_AREA_NAME, CAMERA_INDEX, VIOLATION_THROTTLE_SECONDS
from src.models import Violation, ViolationType, DetectionResult
from src.services.camera_thread import CameraThread
from src.ui.settings_dialog import SettingsDialog

# Giao diện sáng, tông nhẹ nhàng và hơi trong (pastel)
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
import sys
_FONT_FAMILY = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"

_COLORS = {
    "bg_app": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "total_bg": "#FFEBEE",
    "total_accent": "#EB5757",
    "helmet_bg": "#FFF8F3",
    "helmet_accent": "#E67E22",
    "vest_bg": "#FFFBF3",
    "vest_accent": "#D4A017",
    "both_bg": "#FFF3F3",
    "both_accent": "#C0392B",
    "alert_bg": "#FEE8E8",
    "alert_border": "#EB5757",
    "text_primary": "#1E293B",
    "text_secondary": "#64748B",
}


class KPICard(ctk.CTkFrame):
    """Thẻ KPI: nền pastel nhẹ, icon + số màu accent."""

    def __init__(
        self,
        master,
        title: str,
        value: int = 0,
        bg_color: str = "#FFEBEE",
        accent_color: str = "#EB5757",
        icon_text: str = "⚠",
        **kwargs,
    ):
        super().__init__(master, fg_color=bg_color, corner_radius=12, **kwargs)
        self._value_var = ctk.StringVar(value=str(value))
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=12)
        ctk.CTkLabel(
            inner, text=icon_text, font=ctk.CTkFont(_FONT_FAMILY, size=26), text_color=accent_color,
        ).pack(anchor="w")
        ctk.CTkLabel(
            inner, textvariable=self._value_var, font=ctk.CTkFont(_FONT_FAMILY, size=26, weight="bold"), text_color=accent_color,
        ).pack(anchor="w")
        ctk.CTkLabel(
            inner, text=title, font=ctk.CTkFont(_FONT_FAMILY, size=13), text_color=_COLORS["text_secondary"],
        ).pack(anchor="w")

    def set_value(self, value: int):
        self._value_var.set(str(value))


class ViolationCard(ctk.CTkFrame):
    """Thẻ vi phạm: nền pastel nhẹ, viền mỏng."""

    def __init__(self, master, violation: Violation, **kwargs):
        if violation.violation_type == ViolationType.MISSING_BOTH:
            bg_color, accent = _COLORS["both_bg"], _COLORS["both_accent"]
        elif violation.violation_type == ViolationType.MISSING_HELMET:
            bg_color, accent = _COLORS["helmet_bg"], _COLORS["helmet_accent"]
        else:
            bg_color, accent = _COLORS["vest_bg"], _COLORS["vest_accent"]
        super().__init__(master, fg_color=bg_color, corner_radius=10, border_width=1, border_color="#E2E8F0", **kwargs)
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=12, pady=10)
        icon = "⛑🦺" if violation.violation_type == ViolationType.MISSING_BOTH else ("⛑" if violation.violation_type == ViolationType.MISSING_HELMET else "🦺")
        ctk.CTkLabel(inner, text=icon, font=ctk.CTkFont(_FONT_FAMILY, size=20), text_color=accent).pack(side="left", padx=(0, 10))
        right = ctk.CTkFrame(inner, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)
        ctk.CTkLabel(
            right, text=violation.label_vi, font=ctk.CTkFont(_FONT_FAMILY, size=14, weight="bold"), text_color=_COLORS["text_primary"], anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            right, text=f"🕐 {violation.time_str} · Độ chính xác: {violation.confidence_pct}", font=ctk.CTkFont(_FONT_FAMILY, size=12), text_color=_COLORS["text_secondary"], anchor="w",
        ).pack(anchor="w")


class MainWindow:
    """Dashboard giám sát an toàn lao động - nhận diện real-time qua camera."""

    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Hệ thống giám sát an toàn lao động")
        self.root.geometry("1280x780")
        self.root.minsize(1024, 600)
        self.root.configure(fg_color="#F8FAFC")

        # Lưu vi phạm để hiển thị (giới hạn 100)
        self._violations: Deque[Violation] = deque(maxlen=100)
        self._camera_thread: CameraThread | None = None
        self._video_image: ctk.CTkImage | None = None  # Giữ reference để cập nhật
        self._video_label: ctk.CTkLabel | None = None
        self._last_violation_time: dict = {}  # throttle: type -> last add time
        # Nguồn hiện tại: int = camera index, str = đường dẫn file video
        self._current_source: int | str = CAMERA_INDEX

        self._setup_ui()
        self._start_source()

    def _setup_ui(self):
        # ---- Header ----
        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=16)
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", fill="y")
        ctk.CTkLabel(
            left, text="Hệ thống giám sát an toàn lao động",
            font=ctk.CTkFont(_FONT_FAMILY, size=22, weight="bold"), text_color=_COLORS["text_primary"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            left, text="Nhận diện vi phạm trang bị bảo hộ real-time",
            font=ctk.CTkFont(_FONT_FAMILY, size=13), text_color=_COLORS["text_secondary"],
        ).pack(anchor="w")
        ctk.CTkButton(
            header, text="Cài đặt", width=100, font=ctk.CTkFont(_FONT_FAMILY), command=self._on_settings,
        ).pack(side="right", padx=8)

        # ---- KPI cards ----
        kpi_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        kpi_frame.pack(fill="x", padx=24, pady=(0, 12))
        self._kpi_total = KPICard(kpi_frame, "Tổng vi phạm", 0, bg_color=_COLORS["total_bg"], accent_color=_COLORS["total_accent"], icon_text="⚠")
        self._kpi_total.pack(side="left", fill="x", expand=True, padx=4)
        self._kpi_helmet = KPICard(kpi_frame, "Thiếu mũ bảo hộ", 0, bg_color=_COLORS["helmet_bg"], accent_color=_COLORS["helmet_accent"], icon_text="⛑")
        self._kpi_helmet.pack(side="left", fill="x", expand=True, padx=4)
        self._kpi_vest = KPICard(kpi_frame, "Thiếu áo bảo hộ", 0, bg_color=_COLORS["vest_bg"], accent_color=_COLORS["vest_accent"], icon_text="🦺")
        self._kpi_vest.pack(side="left", fill="x", expand=True, padx=4)
        self._kpi_both = KPICard(kpi_frame, "Thiếu cả hai", 0, bg_color=_COLORS["both_bg"], accent_color=_COLORS["both_accent"], icon_text="⛑🦺")
        self._kpi_both.pack(side="left", fill="x", expand=True, padx=4)

        # ---- Alert bar: nền hồng nhạt, viền trái đỏ ----
        self._alert_bar = ctk.CTkFrame(self.root, fg_color=_COLORS["alert_bg"], corner_radius=8, height=48, border_width=0)
        self._alert_bar.pack(fill="x", padx=24, pady=(0, 16))
        # Viền trái đỏ (frame mỏng)
        alert_left_border = ctk.CTkFrame(self._alert_bar, fg_color=_COLORS["alert_border"], width=4, corner_radius=2)
        alert_left_border.place(relx=0, rely=0, relheight=1, anchor="nw")
        inner_alert = ctk.CTkFrame(self._alert_bar, fg_color="transparent")
        inner_alert.place(relx=0.5, rely=0.5, anchor="center")
        self._alert_label = ctk.CTkLabel(
            inner_alert, text="Không có vi phạm.",
            font=ctk.CTkFont(_FONT_FAMILY, size=13), text_color=_COLORS["alert_border"],
        )
        self._alert_label.pack()
        self._alert_bar.pack_forget()  # Ẩn khi không có vi phạm

        self._alert_bar_placeholder = ctk.CTkFrame(self.root, fg_color="transparent", height=8)
        self._alert_bar_placeholder.pack(fill="x")

        # ---- Content: Camera + List ----
        content = ctk.CTkFrame(self.root, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        # Left: Camera
        left_pane = ctk.CTkFrame(content, fg_color="transparent")
        left_pane.pack(side="left", fill="both", expand=True, padx=(0, 12))
        cam_title = ctk.CTkFrame(left_pane, fg_color="transparent")
        cam_title.pack(fill="x", pady=(0, 8))
        self._camera_title_label = ctk.CTkLabel(
            cam_title, text=f"Camera công trường - {CAMERA_AREA_NAME}",
            font=ctk.CTkFont(_FONT_FAMILY, size=15, weight="bold"), text_color=_COLORS["text_primary"],
        )
        self._camera_title_label.pack(side="left")
        ctk.CTkButton(
            cam_title, text="Mở video...", width=100, font=ctk.CTkFont(_FONT_FAMILY),
            command=self._on_open_video,
        ).pack(side="right", padx=4)
        self._btn_back_camera = ctk.CTkButton(
            cam_title, text="Quay lại camera", width=120, font=ctk.CTkFont(_FONT_FAMILY),
            fg_color="gray", command=self._on_back_to_camera,
        )
        self._btn_back_camera.pack(side="right")
        self._btn_back_camera.pack_forget()  # Chỉ hiện khi đang phát video
        self._video_container = ctk.CTkFrame(left_pane, fg_color="#1E293B", corner_radius=10)
        self._video_container.pack(fill="both", expand=True)
        self._video_label = ctk.CTkLabel(
            self._video_container, text="Đang kết nối camera...",
            font=ctk.CTkFont(_FONT_FAMILY, size=14), text_color="gray",
        )
        self._video_label.pack(fill="both", expand=True, padx=8, pady=8)
        self._live_badge = ctk.CTkLabel(
            self._video_container, text=" TRỰC TIẾP ", font=ctk.CTkFont(_FONT_FAMILY, size=11, weight="bold"),
            fg_color="black", text_color="red", corner_radius=4,
        )
        self._live_badge.place(relx=0.02, rely=0.03, anchor="nw")
        self._rec_badge = ctk.CTkLabel(
            self._video_container, text=" REC ", font=ctk.CTkFont(_FONT_FAMILY, size=10),
            fg_color="red", text_color="white", corner_radius=4,
        )
        self._rec_badge.place(relx=0.98, rely=0.03, anchor="ne")

        # Right: Violation list - header nền trắng, bo góc
        right_pane = ctk.CTkFrame(content, fg_color="transparent", width=320)
        right_pane.pack(side="right", fill="y", padx=(12, 0))
        right_pane.pack_propagate(False)
        list_header = ctk.CTkFrame(right_pane, fg_color="white", corner_radius=10, border_width=1, border_color="#E2E8F0")
        list_header.pack(fill="x", pady=(0, 8))
        inner_header = ctk.CTkFrame(list_header, fg_color="transparent")
        inner_header.pack(fill="x", padx=14, pady=10)
        ctk.CTkLabel(inner_header, text="⚠", font=ctk.CTkFont(_FONT_FAMILY, size=18), text_color=_COLORS["total_accent"]).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(inner_header, text="Danh sách vi phạm", font=ctk.CTkFont(_FONT_FAMILY, size=15, weight="bold"), text_color=_COLORS["text_primary"]).pack(side="left")
        self._violation_count_badge = ctk.CTkLabel(
            inner_header, text=" 0 ", font=ctk.CTkFont(_FONT_FAMILY, size=12),
            fg_color=_COLORS["total_accent"], text_color="white", corner_radius=10,
        )
        self._violation_count_badge.pack(side="left", padx=8)
        ctk.CTkButton(
            inner_header, text="Xóa tất cả", fg_color="transparent", text_color=_COLORS["text_secondary"], width=80,
            font=ctk.CTkFont(_FONT_FAMILY), command=self._clear_violations,
        ).pack(side="right")
        self._violation_scroll = ctk.CTkScrollableFrame(right_pane, fg_color="transparent")
        self._violation_scroll.pack(fill="both", expand=True)

    def _on_settings(self):
        """Mở hộp thoại Cài đặt. Sau khi lưu sẽ khởi động lại camera để áp dụng cấu hình mới."""
        def on_saved():
            from src.config.settings import CAMERA_AREA_NAME, CAMERA_INDEX
            if self._camera_thread:
                self._camera_thread.stop()
            self._current_source = CAMERA_INDEX
            self._start_source()
            self._camera_title_label.configure(text=f"Camera công trường - {CAMERA_AREA_NAME}")
            self._btn_back_camera.pack_forget()
            if self._video_label:
                self._video_label.configure(text="")
        d = SettingsDialog(self.root, on_saved=on_saved)
        d.focus_set()

    def _on_open_video(self):
        """Chọn file video và chạy nhận diện trên video."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Chọn video để test nhận diện",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("MP4", "*.mp4"),
                ("Tất cả", "*.*"),
            ],
        )
        if not path:
            return
        if self._camera_thread:
            self._camera_thread.stop()
        self._current_source = path
        self._start_source()
        import os
        self._camera_title_label.configure(text=f"Video: {os.path.basename(path)}")
        self._btn_back_camera.pack(side="right", padx=4)
        self._video_label.configure(text="")

    def _on_back_to_camera(self):
        """Chuyển từ video về camera trực tiếp."""
        from src.config.settings import CAMERA_INDEX, CAMERA_AREA_NAME
        if self._camera_thread:
            self._camera_thread.stop()
        self._current_source = CAMERA_INDEX
        self._start_source()
        self._camera_title_label.configure(text=f"Camera công trường - {CAMERA_AREA_NAME}")
        self._btn_back_camera.pack_forget()
        self._video_label.configure(text="")

    def _on_video_ended(self):
        """Gọi khi video file phát xong."""
        if self._video_label:
            self.root.after(0, lambda: self._video_label.configure(text="Video kết thúc."))

    def _clear_violations(self):
        self._violations.clear()
        self._refresh_violation_list()
        self._update_kpis(0, 0, 0, 0)
        self._update_alert(0)

    def _start_camera(self):
        """Khởi động lại camera (gọi từ on_saved trong Cài đặt)."""
        from src.config.settings import CAMERA_INDEX
        self._current_source = CAMERA_INDEX
        self._start_source()

    def _start_source(self):
        """Khởi động nguồn hiện tại (camera hoặc file video)."""
        self._camera_thread = CameraThread(
            on_frame=self._on_frame,
            source=self._current_source,
            on_video_end=self._on_video_ended,
        )
        self._camera_thread.start()

    def _on_frame(self, result: DetectionResult):
        """Callback từ camera thread: throttle vi phạm rồi gửi lên main thread."""
        import time
        now = time.time()
        to_add: List[Violation] = []
        for v in result.violations:
            key = v.violation_type.value
            if now - self._last_violation_time.get(key, 0) >= VIOLATION_THROTTLE_SECONDS:
                to_add.append(v)
                self._last_violation_time[key] = now
        self.root.after(0, lambda: self._on_frame_ui(result, to_add))

    def _on_frame_ui(self, result: DetectionResult, to_add: List[Violation]):
        """Chạy trên main thread: cập nhật ảnh, thêm vi phạm, refresh UI."""
        for v in to_add:
            self._violations.append(v)
        # Cập nhật ảnh video
        frame = result.frame
        if frame is not None and frame.size > 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            max_w, max_h = 640, 480
            scale = min(max_w / w, max_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(frame_rgb).resize((new_w, new_h), PILImage.Resampling.LANCZOS)
                self._set_video_image(pil_img, new_w, new_h)
        total = len(self._violations)
        h_count = sum(1 for x in self._violations if x.violation_type == ViolationType.MISSING_HELMET)
        v_count = sum(1 for x in self._violations if x.violation_type == ViolationType.MISSING_VEST)
        b_count = sum(1 for x in self._violations if x.violation_type == ViolationType.MISSING_BOTH)
        self._update_kpis(total, h_count, v_count, b_count)
        if to_add:
            self._refresh_violation_list()
        self._update_alert(total)

    def _set_video_image(self, pil_img: "Image.Image", w: int, h: int):
        if self._video_label is None:
            return
        # CTkImage yêu cầu light_image và dark_image cùng kích thước; khi đổi nguồn (camera ↔ video) kích thước frame có thể đổi → tạo mới CTkImage thay vì configure.
        current_size = getattr(self, "_video_image_size", None)
        if self._video_image is None or current_size != (w, h):
            self._video_image_size = (w, h)
            self._video_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))
            self._video_label.configure(image=self._video_image, text="")
        else:
            self._video_image.configure(light_image=pil_img, dark_image=pil_img, size=(w, h))
            self._video_label.configure(image=self._video_image, text="")

    def _update_kpis(self, total: int, helmet: int, vest: int, both: int):
        self._kpi_total.set_value(total)
        self._kpi_helmet.set_value(helmet)
        self._kpi_vest.set_value(vest)
        self._kpi_both.set_value(both)

    def _update_alert(self, total: int):
        if total > 0:
            self._alert_bar_placeholder.pack_forget()
            self._alert_bar.pack(fill="x", padx=24, pady=(0, 16))
            self._alert_label.configure(
                text=f"Cảnh báo! Phát hiện {total} vi phạm an toàn lao động. Vui lòng kiểm tra và xử lý ngay lập tức."
            )
        else:
            self._alert_bar.pack_forget()
            self._alert_bar_placeholder.pack(fill="x")

    def _refresh_violation_list(self):
        for w in self._violation_scroll.winfo_children():
            w.destroy()
        for v in reversed(list(self._violations)):
            card = ViolationCard(self._violation_scroll, v)
            card.pack(fill="x", pady=4)
        self._violation_count_badge.configure(text=f" {len(self._violations)} ")

    def run(self):
        try:
            self.root.mainloop()
        finally:
            if self._camera_thread:
                self._camera_thread.stop()
