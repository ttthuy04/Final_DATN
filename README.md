# Student Behavior Recognition

Hệ thống **nhận diện hành vi sinh viên trong lớp học** từ video, dựa trên **skeleton (khung xương)** thay vì xử lý pixel thô. Dự án hỗ trợ đầy đủ quy trình: chuẩn bị dữ liệu → trích xuất đặc trưng → huấn luyện mô hình → đánh giá → triển khai demo (CLI và Web).

---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
3. [Công nghệ sử dụng](#công-nghệ-sử-dụng)
4. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
5. [Cài đặt môi trường](#cài-đặt-môi-trường)
6. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
7. [Trích xuất skeleton](#trích-xuất-skeleton)
8. [Huấn luyện mô hình](#huấn-luyện-mô-hình)
9. [Đánh giá mô hình](#đánh-giá-mô-hình)
10. [Demo nhận diện](#demo-nhận-diện)
11. [Ứng dụng Web](#ứng-dụng-web)
12. [Đầu ra của từng bước](#đầu-ra-của-từng-bước)
13. [Lưu ý & xử lý sự cố](#lưu-ý--xử-lý-sự-cố)

---

## Tổng quan

Hệ thống phân loại **4 hành vi** của sinh viên trong lớp:

| Nhãn (thư mục) | Mô tả |
|----------------|--------|
| `normal` | Ngồi học bình thường |
| `sleep` | Ngủ gật |
| `standing` | Đứng dậy |
| `use_phone` | Dùng điện thoại |

**Luồng xử lý chính:**

```text
Video gốc → Cắt & gán nhãn → Trích xuất skeleton → Train model → Evaluate → Demo/Web
```

**Hai mô hình phân loại** (cùng input skeleton):

| Mô hình | Thư mục | Vai trò |
|---------|---------|---------|
| **LSTM** | `src/lstm/` | Baseline, so sánh thực nghiệm |
| **ST-GCN** | `src/stgcn/` | Mô hình chính, dùng trong demo & web |

---

## Kiến trúc hệ thống

### Pipeline huấn luyện (offline)

```text
data/raw/              Video lớp học dài
    ↓  cut_video.py (thủ công: chọn ROI + nhãn)
data/labeled/          Clip .mp4 theo từng hành vi (1 người/clip)
    ↓  extract_skeleton.py (MediaPipe Pose)
data/skeleton/         File .npy — mỗi clip = ma trận (60, 100)
    ↓  skeleton_dataset.py (chia 70/15/15)
train.py               Huấn luyện LSTM hoặc ST-GCN
    ↓
models_saved/          Trọng số best.pt
results/               Log, biểu đồ, metrics
```

### Pipeline inference (demo / web)

```text
Video lớp (nhiều người)
    ↓  YOLOv8 — phát hiện & theo dõi người
    ↓  Crop từng bbox
    ↓  MediaPipe — skeleton trên crop
    ↓  Buffer 60 frame / track
    ↓  ST-GCN — dự đoán hành vi
Video kết quả + thống kê JSON
```

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ | Python 3.10+ (khuyến nghị) |
| Xử lý video | OpenCV |
| Phát hiện người | YOLOv8 (`ultralytics`) |
| Trích xuất skeleton | MediaPipe Pose Landmarker |
| Deep learning | PyTorch |
| Mô hình 1 | LSTM 2 chiều |
| Mô hình 2 | ST-GCN (Yan et al., AAAI 2018) |
| Chia dữ liệu / metrics | scikit-learn |
| **Backend Web** | Flask (Python) |
| **Frontend Web** | HTML + CSS + JavaScript (Vanilla JS, `fetch` API) |
| Chuyển mã video web | imageio-ffmpeg (H.264) |

---

## Cấu trúc thư mục

```text
Student_Behavior_Recognition_v3/
├── data/                        # Dataset (LOCAL — không có trên GitHub)
│   ├── raw/                     # Video gốc từ camera/lớp học
│   ├── labeled/                 # Clip đã cắt + gán nhãn
│   │   ├── normal/
│   │   ├── sleep/
│   │   ├── standing/
│   │   └── use_phone/
│   └── skeleton/                # Skeleton .npy (đầu ra trích xuất)
├── models/                      # Model MediaPipe (tự tải khi chạy lần đầu)
├── models_saved/                # Trọng số sau train (gitignore)
│   ├── lstm/best.pt
│   └── stgcn/best.pt
├── results/                     # Kết quả train, evaluate, demo
│   ├── lstm/
│   ├── stgcn/
│   └── demo_check/              # Kết quả demo CLI / Web
├── src/
│   ├── common/                  # Paths, log, plot, YOLO filter, demo presets
│   ├── dataset/                 # Loader skeleton, chia train/val/test
│   ├── demo/                    # Pipeline demo lớp học
│   ├── lstm/                    # BiLSTM: model, train, evaluate
│   ├── preprocess/              # Cắt video, extract skeleton
│   ├── stgcn/                   # ST-GCN: model, train, evaluate
│   └── visualization/           # Vẽ khung xương lên frame
├── script/                      # Script kiểm tra từng bước (step1, step2, step3)
├── web/                         # Ứng dụng web (Flask + index.html)
│   ├── server.py
│   └── index.html
├── requirements.txt
└── README.md
```

Chi tiết cấu trúc `data/` xem thêm tại [`data/README.md`](data/README.md).

---

## Cài đặt môi trường

### Yêu cầu phần cứng & phần mềm

- **OS:** Windows 10/11, Linux hoặc macOS
- **Python:** 3.10 trở lên
- **RAM:** Tối thiểu 8 GB (khuyến nghị 16 GB khi train/demo)
- **GPU:** Không bắt buộc; có CUDA sẽ train/nhanh hơn
- **Dung lượng:** ~2 GB cho môi trường Python; dataset video riêng (~50 GB, lưu local)

### Bước 1 — Clone repository

```powershell
git clone <url-repo>
cd Student_Behavior_Recognition_v3
```

### Bước 2 — Tạo virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 3 — Cài đặt dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Các gói chính trong `requirements.txt`:**

| Gói | Mục đích |
|-----|----------|
| `opencv-python` | Đọc/ghi video, hiển thị |
| `mediapipe` | Trích xuất pose skeleton |
| `torch` | Huấn luyện & inference deep learning |
| `ultralytics` | YOLOv8 phát hiện người |
| `scikit-learn` | Chia train/val/test, metrics |
| `flask` | Web server |
| `imageio-ffmpeg` | Chuyển video cho trình duyệt |

### Bước 4 — (Tùy chọn) Cài PyTorch có GPU

Nếu máy có NVIDIA GPU, có thể cài bản CUDA phù hợp từ [pytorch.org](https://pytorch.org) thay vì bản CPU mặc định.

### Bước 5 — Tạo cấu trúc thư mục dữ liệu

```powershell
mkdir data\raw, data\labeled, data\skeleton
```

Đặt ít nhất một video thử vào `data/raw/` (ví dụ `test1.mp4`).

### Bước 6 — Kiểm tra cài đặt

```powershell
python -c "import cv2, mediapipe, torch, ultralytics, flask; print('OK')"
```

---

## Chuẩn bị dữ liệu

> **Lưu ý:** Thư mục `data/` **không được đẩy lên GitHub** (~50 GB). Mỗi máy clone repo cần tự chuẩn bị dữ liệu local.

### Bước 1 — Đặt video gốc

Copy video quay lớp học (file `.mp4`) vào:

```text
data/raw/
```

Ví dụ: `data/raw/video_demo.mp4`, `data/raw/test1.mp4`

### Bước 2 — Cắt video và gán nhãn (thủ công)

**File:** `src/preprocess/cut_video.py`

Công cụ tương tác: mở video dài, chọn vùng sinh viên (ROI), ghi clip ngắn và gán nhãn hành vi.

**Chạy:**

```powershell
python src/preprocess/cut_video.py --video data/raw/video_demo.mp4 --save_dir data/labeled
```

**Phím tắt trên cửa sổ video:**

| Phím | Chức năng |
|------|-----------|
| `s` | Dừng, kéo chuột khoanh vùng sinh viên, bắt đầu ghi |
| `e` | Kết thúc ghi clip (sau đó nhập nhãn trên terminal) |
| `f` / `b` | Tua tới / lùi 5 giây |
| `1` / `2` / `3` | Tốc độ phát 1x / 2x / 4x |
| `q` | Thoát |

**Sau khi nhấn `e`:** nhập nhãn trên terminal, ví dụ `sleep`, `use_phone`, `normal`, `standing`.

**Input:**

| Tham số | Mặc định | Mô tả |
|---------|----------|--------|
| `--video` | `data/raw/video_demo.mp4` | Video gốc dài |
| `--save_dir` | `data/labeled` | Thư mục lưu clip |

**Output:**

```text
data/labeled/
├── normal/normal_1.mp4, normal_2.mp4, ...
├── sleep/sleep_1.mp4, ...
├── standing/standing_1.mp4, ...
└── use_phone/use_phone_1.mp4, ...
```

Mỗi file là **một clip ngắn, một sinh viên, một hành vi** — đã crop theo ROI bạn chọn.

---

## Trích xuất skeleton

**File:** `src/preprocess/extract_skeleton.py`

Chuyển mỗi clip `.mp4` trong `data/labeled/` thành file số `.npy` mô tả tư thế theo thời gian, dùng **MediaPipe Pose Landmarker**.

**Chạy (xử lý toàn bộ dataset):**

```powershell
python src/preprocess/extract_skeleton.py
```

**Tùy chọn:**

```powershell
# Ghi đè file .npy đã có
python src/preprocess/extract_skeleton.py --overwrite

# Chỉ xử lý một số nhãn
python src/preprocess/extract_skeleton.py --labels sleep,use_phone

# Bỏ qua nhãn (mặc định bỏ qua "distracted" nếu có)
python src/preprocess/extract_skeleton.py --skip_labels distracted

# Đổi số frame mỗi video (mặc định 60)
python src/preprocess/extract_skeleton.py --n_frames 60
```

### Cách trích xuất hoạt động

1. **Lấy mẫu 60 frame đều** từ mỗi video (skip frame để phủ toàn bộ độ dài clip).
2. Mỗi frame đưa vào MediaPipe → phát hiện **25 landmark phần thân trên**.
3. Mỗi landmark → 4 giá trị: `(x, y, z, visibility)`.
4. Gộp thành vector **100 số / frame** (25 × 4).
5. Nếu frame không detect được pose → lặp frame trước hoặc điền 0.

### Định dạng dữ liệu skeleton

```text
Shape mỗi file .npy: (60, 100)

  60  = số frame thời gian (chuỗi hành vi ngắn)
  100 = 25 khớp × 4 (x, y, z, visibility)
```

**Input:**

| Nguồn | Mô tả |
|-------|--------|
| `data/labeled/<nhãn>/*.mp4` | Clip đã gán nhãn |
| `models/pose_landmarker_lite.task` | Tự tải lần đầu chạy |

**Output:**

```text
data/skeleton/
├── normal/normal_1.npy
├── sleep/sleep_1.npy
├── standing/standing_1.npy
└── use_phone/use_phone_1.npy
```

### Kiểm tra dữ liệu đã load đúng

```powershell
python src/dataset/skeleton_dataset.py
```

In ra số mẫu train/val/test và shape batch `(32, 60, 100)`.

---

## Huấn luyện mô hình

Sau khi có `data/skeleton/`, module `src/dataset/skeleton_dataset.py` sẽ:

- Đọc tất cả `.npy` → `X` shape `(N, 60, 100)`, `y` shape `(N,)`
- Chia **70% train / 15% val / 15% test** (stratified, `random_state=42`)

### Huấn luyện BiLSTM

```powershell
python src/lstm/train.py
```

- Kiến trúc: BiLSTM 2 lớp, hidden=128, dropout=0.3 → Linear
- Optimizer: Adam + ReduceLROnPlateau + early stopping
- Class weights: cân bằng `normal`, `use_phone`

**Output:**

```text
models_saved/lstm/best.pt
results/lstm/training_log.csv
```

### Huấn luyện ST-GCN (mô hình chính)

```powershell
python src/stgcn/train.py
```

- Graph convolution trên 25 khớp + trục thời gian
- Root normalization (tâm vai), velocity stream
- Spatial configuration partitioning (Yan et al. 2018)

**Output:**

```text
models_saved/stgcn/best.pt
results/stgcn/training_log.csv
```

### Vẽ biểu đồ quá trình train

```powershell
python src/common/plot_history.py --model lstm
python src/common/plot_history.py --model stgcn
```

Tạo `results/<model>/training_history.png`.

---

## Đánh giá mô hình

Chạy trên **test set** (15% dữ liệu, không dùng khi train):

```powershell
python src/lstm/evaluate.py
python src/stgcn/evaluate.py
```

**Metrics:** Accuracy, Precision, Recall, F1 (macro), classification report.

**Output mỗi model:**

| File | Mô tả |
|------|--------|
| `results/<model>/test_metrics.json` | Metrics tổng hợp |
| `results/<model>/confusion_matrix.png` | Ma trận nhầm lẫn |
| `results/<model>/training_log.csv` | Log từng epoch |
| `results/<model>/training_history.png` | Biểu đồ loss/accuracy |

---

## Demo nhận diện

Demo chạy trên **video lớp đầy đủ** (nhiều sinh viên), khác với dữ liệu train (clip 1 người).

**Pipeline:** YOLOv8m → crop người → MediaPipe skeleton → ST-GCN → video + thống kê

### Kiểm tra từng bước (tùy chọn)

```powershell
# Bước 1: Kiểm tra YOLO detect người
python script/step1_yolo_detect.py --video data/raw/test1.mp4

# Bước 2: Kiểm tra skeleton trên crop
python script/step2_mediapipe_crop.py --video data/raw/test1.mp4

# Bước 3: Demo đầy đủ
python script/step3_classroom_demo.py --video data/raw/test1.mp4
```

**Tùy chọn step3:**

```powershell
# Giới hạn số frame (test nhanh)
python script/step3_classroom_demo.py --video data/raw/test1.mp4 --max-frames 300

# Không vẽ khung xương lên video
python script/step3_classroom_demo.py --video data/raw/test1.mp4 --no-skeleton
```

**Output demo:**

```text
results/demo_check/<ten_video>_<timestamp>/
├── input.mp4                  # Video đầu vào (bản sao)
├── classroom_demo_final.mp4   # Video kết quả (OpenCV)
├── classroom_demo_final_web.mp4  # Bản H.264 cho trình duyệt
└── result.json                # Thống kê hành vi, số track, v.v.
```

> Demo yêu cầu đã train ST-GCN: `models_saved/stgcn/best.pt` phải tồn tại.

---

## Ứng dụng Web

Giao diện web cho phép upload video và xem kết quả phân tích qua trình duyệt, không cần chạy lệnh terminal.

| Thành phần | Công nghệ |
|------------|-----------|
| Frontend | HTML + CSS + JavaScript (Vanilla JS) |
| Backend | Flask (`web/server.py`) |
| API | REST (`/api/analyze`, `/api/jobs/...`, `/api/runs`) |

### Khởi chạy

```powershell
python web/server.py
```

Mở trình duyệt: **http://127.0.0.1:5050**

### Luồng hoạt động

1. Người dùng chọn file video trên giao diện.
2. Frontend gửi `POST /api/analyze` (multipart form).
3. Flask lưu video, tạo job, chạy `run_classroom_demo()` trong thread nền.
4. Frontend poll `GET /api/jobs/<job_id>` để hiển thị tiến trình.
5. Khi xong: trả video kết quả + bảng thống kê; lưu tại `results/demo_check/`.

### API chính

| Endpoint | Method | Mô tả |
|----------|--------|--------|
| `/` | GET | Trang chủ (index.html) |
| `/api/health` | GET | Kiểm tra server |
| `/api/analyze` | POST | Upload video, bắt đầu phân tích |
| `/api/jobs/<job_id>` | GET | Trạng thái job (progress %) |
| `/api/runs` | GET | Danh sách lần chạy trước |
| `/api/runs/<run_name>` | GET | Chi tiết kết quả một lần chạy |
| `/files/<run_name>/<file>` | GET | Tải video / file kết quả |

---

## Đầu ra của từng bước

| Bước | Input | Output |
|------|--------|--------|
| Cắt video | Video dài `.mp4` + nhãn thủ công | `data/labeled/<nhãn>/*.mp4` |
| Trích xuất skeleton | Clip `.mp4` | `.npy` shape `(60, 100)` |
| Load dataset | Nhiều `.npy` | `X: (N,60,100)`, `y: (N,)` |
| Train | Batch skeleton + nhãn | `models_saved/<model>/best.pt` |
| Evaluate | Test loader + `best.pt` | JSON metrics, confusion matrix |
| Demo / Web | Video lớp full frame | Video annotated + `result.json` |

---

## Lưu ý & xử lý sự cố

### Dataset không có sau khi clone

```text
FileNotFoundError: Khong tim thay: data/labeled
```

→ Tạo thư mục `data/` và chạy `cut_video.py` hoặc copy dữ liệu từ máy khác.

### Chưa có model khi chạy demo/web

```text
FileNotFoundError: Chua co ST-GCN tai models_saved/stgcn/best.pt
```

→ Chạy `python src/stgcn/train.py` trước (cần có `data/skeleton/`).

### MediaPipe model tự tải

Lần đầu chạy `extract_skeleton.py`, model `pose_landmarker_lite.task` được tải vào `models/`. Cần kết nối internet.

### Video không phát trên trình duyệt

Web server tự chuyển sang H.264 qua `imageio-ffmpeg`. Nếu thiếu ffmpeg, cài lại: `pip install imageio-ffmpeg`.

### Train chậm trên CPU

Bình thường. Có GPU NVIDIA thì cài PyTorch CUDA để tăng tốc.

### YOLO tải weight lần đầu

`yolov8m.pt` được Ultralytics tự tải khi chạy demo lần đầu.

---

## Tóm tắt lệnh nhanh

```powershell
# 1. Cài đặt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Chuẩn bị dữ liệu
python src/preprocess/cut_video.py --video data/raw/video_demo.mp4
python src/preprocess/extract_skeleton.py

# 3. Train & đánh giá (ST-GCN)
python src/stgcn/train.py
python src/stgcn/evaluate.py
python src/common/plot_history.py --model stgcn

# 4. Demo CLI
python script/step3_classroom_demo.py --video data/raw/test1.mp4

# 5. Web
python web/server.py
```

---

## Tài liệu tham khảo

- ST-GCN: Yan et al., *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*, AAAI 2018
- MediaPipe Pose: [Google MediaPipe](https://developers.google.com/mediapipe)
- YOLOv8: [Ultralytics](https://docs.ultralytics.com)
