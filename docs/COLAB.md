# Chạy training trên Google Colab

Ứng dụng giám sát (giao diện desktop) chạy trên máy bạn. **Colab dùng để train model YOLO** (và chạy Data Augmentation) nhờ GPU miễn phí, sau đó tải file `best.pt` về và dùng trong app.

## Cách 1: Mở notebook có sẵn trong dự án

1. Vào [Google Colab](https://colab.research.google.com/).
2. **File → Upload notebook** → chọn file `notebooks/train_on_colab.ipynb` trong dự án.
3. Hoặc đẩy dự án lên GitHub, rồi trong Colab: **File → Open notebook → GitHub** → dán URL repo → chọn `notebooks/train_on_colab.ipynb`.

## Cách 2: Tạo notebook mới và copy từng bước

Tạo notebook mới trên Colab, bật **GPU**: **Runtime → Change runtime type → Hardware accelerator: GPU**.

### Bước 1: Đưa dự án vào Colab

**Cách A — Clone từ GitHub (nếu đã có repo):**

```python
!git clone https://github.com/YOUR_USERNAME/Contruction-Site-Safety.git /content/Contruction
%cd /content/Contruction
```

**Cách B — Upload ZIP dự án:**

- Nén cả thư mục dự án (chứa `src`, `notebooks`, …) thành một file `.zip`.
- Trong Colab chạy:

```python
from google.colab import files
import zipfile, os
uploaded = files.upload()  # Chọn file .zip
for name in uploaded:
    if name.endswith('.zip'):
        with zipfile.ZipFile(name, 'r') as z:
            z.extractall('/content')
        break
%cd /content/Contruction   # hoặc %cd /content/TÊN_THƯ_MỤC nếu giải nén ra tên khác
```

### Bước 2: Cài đặt thư viện

```python
!pip install -q ultralytics opencv-python numpy Pillow
import sys
sys.path.insert(0, '/content/Contruction')
```

### Bước 3: Đưa dataset lên Colab

Cấu trúc YOLO chuẩn:

- `train/images/` — ảnh train  
- `train/labels/` — file `.txt` label (mỗi dòng: `class_id x_center y_center width height`, normalized 0–1)  
- `val/images/` và `val/labels/` — validation  

Nén thành một file ZIP (ví dụ `dataset.zip`), trong Colab:

```python
from google.colab import files
import zipfile, os
uploaded = files.upload()
for name in uploaded:
    if name.endswith('.zip'):
        os.makedirs('/content/dataset', exist_ok=True)
        with zipfile.ZipFile(name, 'r') as z:
            z.extractall('/content/dataset')
        break
```

### Bước 4 (tùy chọn): Data Augmentation

Chỉ chạy nếu đã có `train/images` và `train/labels`:

```python
import os, sys, subprocess
sys.path.insert(0, '/content/Contruction')
train_img = '/content/dataset/train/images'
train_lbl = '/content/dataset/train/labels'
out_aug = '/content/dataset/train_aug'
if os.path.isdir(train_img) and os.path.isdir(train_lbl):
    subprocess.run([sys.executable, '-m', 'src.training.prepare_augmented_dataset',
        '--images', train_img, '--labels', train_lbl, '--out', out_aug,
        '--num', '2', '--copy-original'], cwd='/content/Contruction', check=True)
```

Nếu dùng dữ liệu đã augment, trong bước 5 đổi `train: train/images` thành `train: train_aug/images` trong `data.yaml`.

### Bước 5: Tạo data.yaml và train

```python
yaml_content = """
path: /content/dataset
train: train/images
val: val/images
names:
  0: Gloves
  1: Helmet
  2: Non-Helmet
  3: Person
  4: Shoes
  5: Vest
  6: bare-arms
"""
with open('/content/Contruction/data_colab.yaml', 'w') as f:
    f.write(yaml_content.strip())

from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.train(
    data='/content/Contruction/data_colab.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project='/content/Contruction/runs',
    name='ppe_detect',
    exist_ok=True,
)
```

### Bước 6: Tải model về máy

```python
from google.colab import files
import os
best = '/content/Contruction/runs/ppe_detect/weights/best.pt'
if os.path.isfile(best):
    files.download(best)
```

Đặt file `best.pt` vào thư mục dự án (hoặc trỏ **Cài đặt → Đường dẫn model** trong app tới file này).

---

**Lưu ý:** Nếu dùng **Google Drive** để lưu dataset hoặc dự án, mount Drive rồi đổi các đường dẫn `/content/...` thành đường dẫn trong Drive (ví dụ `/content/drive/MyDrive/Contruction/...`).
