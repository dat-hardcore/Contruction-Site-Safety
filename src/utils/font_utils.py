"""
Font hỗ trợ tiếng Việt để vẽ lên ảnh (PIL).
"""
import os
from typing import Optional

from PIL import ImageFont


def get_vietnamese_font(size: int = 14) -> Optional[ImageFont.FreeTypeFont]:
    """
    Trả về font PIL hỗ trợ tiếng Việt, hoặc None nếu không tìm thấy.
    Ưu tiên: Segoe UI (Windows), Arial, DejaVu Sans.
    """
    candidates = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        candidates = [
            os.path.join(windir, "Fonts", "segoeui.ttf"),
            os.path.join(windir, "Fonts", "arial.ttf"),
            os.path.join(windir, "Fonts", "tahoma.ttf"),
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None
