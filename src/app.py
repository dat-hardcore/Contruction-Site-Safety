"""
Lớp ứng dụng - khởi tạo và chạy app.
"""
from src.ui import MainWindow


class Application:
    """Điểm điều khiển chính của ứng dụng."""

    def __init__(self):
        self.window = MainWindow()

    def run(self):
        """Chạy ứng dụng."""
        self.window.run()
