"""
Cấu hình mặc định.
Có thể ghi đè bằng file data/settings.json (tạo qua màn hình Cài đặt).
"""
import json
import os

# Đường dẫn gốc dự án (thư mục chứa src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Assets
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
ICONS_DIR = os.path.join(ASSETS_DIR, "icons")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")

# File cài đặt do người dùng lưu (nếu có)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
USER_SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

# Camera & detection (giá trị mặc định)
CAMERA_INDEX = 0
CAMERA_AREA_NAME = "Khu vực A"
CONFIDENCE_THRESHOLD = 0.25
VIOLATION_THROTTLE_SECONDS = 0.8
MODEL_PATH = os.path.join(PROJECT_ROOT, "helmet_best.pt")
YOLO26_MODEL = "yolo26n.pt"
# Chạy detection mỗi N frame (dùng chung cho cả camera và video); giữa các lần track Person hoặc vẽ lại box.
DETECT_EVERY_N_FRAMES = 5
# Kích thước ảnh đưa vào YOLO — dùng cho cả camera và video (nhỏ = nhanh hơn).
DETECT_IMGSZ = 416
# Chỉ track box "Person" giữa các lần detect (cả camera và video), ít tracker = ít lag.
TRACK_PERSON_ONLY = True

# Ghi hình
RECORDING_ENABLED = False
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "data", "recordings")


def load_user_settings():
    """Đọc data/settings.json (nếu có) và ghi đè lên biến cấu hình."""
    global CAMERA_INDEX, CAMERA_AREA_NAME, CONFIDENCE_THRESHOLD
    global VIOLATION_THROTTLE_SECONDS, MODEL_PATH
    if not os.path.isfile(USER_SETTINGS_PATH):
        return
    try:
        with open(USER_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "camera_index" in data:
            CAMERA_INDEX = int(data["camera_index"])
        if "camera_area_name" in data:
            CAMERA_AREA_NAME = str(data["camera_area_name"]).strip()
        if "confidence_threshold" in data:
            CONFIDENCE_THRESHOLD = float(data["confidence_threshold"])
        if "violation_throttle_seconds" in data:
            VIOLATION_THROTTLE_SECONDS = float(data["violation_throttle_seconds"])
        if "model_path" in data and data["model_path"]:
            MODEL_PATH = str(data["model_path"]).strip()
    except Exception:
        pass


# Áp dụng cài đặt người dùng khi import
load_user_settings()
