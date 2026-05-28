# Student Behavior Recognition

Hệ thống nhận diện hành vi sinh viên trong lớp học dựa trên video, sử dụng pipeline: tiền xử lý video → trích xuất skeleton → huấn luyện mô hình (LSTM / ST-GCN / ST-LSTM).

## Cấu trúc dự án

```
Student_Behavior_Recognition_v3/
├── data/                    # Dataset (LOCAL — không có trên GitHub)
│   ├── raw/                 # Video gốc
│   ├── labeled/             # Clip đã gán nhãn
│   └── skeleton/            # Dữ liệu skeleton
├── src/
│   ├── dataset/             # Dataset loader, build dataset
│   ├── models/              # LSTM, ST-GCN, ST-LSTM
│   ├── preprocess/          # Cắt video, trích xuất skeleton
│   ├── train/               # Script huấn luyện
│   └── visualization/       # Hiển thị skeleton
├── main.py
├── requirements.txt
└── README.md
```

## Nhãn hành vi

| Thư mục | Mô tả |
|---------|--------|
| `distracted` | Mất tập trung |
| `normal` | Ngồi học bình thường |
| `sleep` | Ngủ gật |
| `standing` | Đứng dậy |
| `use_phone` | Dùng điện thoại |

## Cài đặt

```bash
# Clone repository
git clone https://github.com/<username>/<repo-name>.git
cd Student_Behavior_Recognition_v3

# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

## Dataset

**Dữ liệu không nằm trên GitHub** (~50 GB video/nhãn).

Sau khi clone, tạo hoặc copy dữ liệu vào `data/` theo hướng dẫn trong [`data/README.md`](data/README.md):

1. `data/raw/` — video gốc  
2. `data/labeled/<class>/` — clip đã gán nhãn  
3. `data/skeleton/` — đầu ra bước trích xuất skeleton  

Sao lưu dataset: ổ cứng ngoài / cloud (OneDrive, Google Drive, …). Có thể ghi link tải dataset trong Issues hoặc tài liệu đồ án.

## Quy trình làm việc

```text
raw video  →  cut & label  →  extract skeleton  →  train model  →  evaluate / visualize
   (raw)         (labeled)         (skeleton)          (models)
```

Script tham chiếu:

- `src/preprocess/cut_video.py` — cắt video  
- `src/preprocess/extract_skeleton.py` — trích xuất skeleton  
- `src/dataset/build_dataset.py` — xây dataset  
- `src/models/` — định nghĩa mô hình  
- `src/visualization/visualize_skeleton.py` — kiểm tra skeleton  

## Đẩy code lên GitHub

```bash
git init
git add .
git status    # Đảm bảo KHÔNG có file trong data/raw, data/labeled, data/skeleton
git commit -m "Initial commit: project structure and source code"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

Trước `git add`, luôn chạy `git status` — chỉ thấy code, `README.md`, `.gitignore`, và `data/README.md`.

