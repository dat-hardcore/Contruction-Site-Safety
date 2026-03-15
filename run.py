#!/usr/bin/env python3
"""
Script chạy nhanh từ thư mục gốc: python run.py
"""
import sys
import os

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main

if __name__ == "__main__":
    sys.exit(main())
