# Construction Site Safety

Hệ thống giám sát an toàn lao động – nhận diện vi phạm trang bị bảo hộ (mũ, áo) real-time qua camera.

## Tính năng

- **Dashboard**: Header, 4 thẻ KPI (Tổng vi phạm, Thiếu mũ bảo hộ, Thiếu áo bảo hộ, Thiếu cả hai), thanh cảnh báo.
- **Camera trực tiếp**: Hiển thị video từ camera với overlay bounding box vi phạm (TRỰC TIẾP, REC).
- **Danh sách vi phạm**: Cuộn được, từng dòng có loại vi phạm, thời gian, độ chính xác; nút "Xóa tất cả".
- **Nhận diện real-time**: Mô-đun phát hiện PPE (mũ/áo). Chưa có file model thì chạy chế độ mẫu; có file YOLO `.pt` thì dùng model thật.

## Cấu trúc thư mục

```
Contruction/
├── src/
│   ├── main.py
│   ├── app.py
│   ├── ui/
│   │   └── main_window.py   # Dashboard (CustomTkinter)
│   ├── models/
│   │   └── violation.py     # Violation, ViolationType, DetectionResult
│   ├── services/
│   │   ├── detector.py      # PPEDetector (YOLO hoặc demo)
│   │   └── camera_thread.py # Luồng camera + detection
│   └── config/
│       └── settings.py      # CAMERA_INDEX, MODEL_PATH, ...
├── assets/
├── data/
├── requirements.txt
├── run.py
└── README.md
```

## Cài đặt và chạy

```bash
cd "d:\book\HAU\Contruction Site Safety\Contruction"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Dùng model YOLO thật

1. Huấn luyện hoặc tải model YOLO phát hiện PPE (ví dụ dataset "Hard Hat", "Safety Vest" trên Roboflow).
2. Trong `src/config/settings.py` đặt `MODEL_PATH = r"đường_dẫn\tới\model.pt"`.
3. Trong `src/services/detector.py` chỉnh `CLASS_NAMES` và `CLASS_TO_VIOLATION` cho đúng nhãn dataset của bạn (ví dụ: 0=with_helmet, 1=without_helmet, 2=with_vest, 3=without_vest).

## Train model trên Google Colab

Để train YOLO (và chạy Data Augmentation) trên Colab (GPU miễn phí): mở notebook `notebooks/train_on_colab.ipynb` trong [Google Colab](https://colab.research.google.com/) (Upload notebook hoặc mở từ GitHub). Chi tiết từng bước: [docs/COLAB.md](docs/COLAB.md).

## Cấu hình

- `src/config/settings.py`: `CAMERA_INDEX` (0 = webcam), `CAMERA_AREA_NAME`, `CONFIDENCE_THRESHOLD`, `MODEL_PATH`.
