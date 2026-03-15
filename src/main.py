"""
Điểm vào chính của ứng dụng.
"""
import sys


def main():
    """Entry point."""
    from src.app import Application

    app = Application()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
