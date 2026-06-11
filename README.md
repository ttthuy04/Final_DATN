# Student Behavior Recognition

Hệ thống nhận diện hành vi sinh viên trong lớp học dựa trên video.

**Pipeline:** tiền xử lý video → trích xuất skeleton → train model → evaluate

## Cấu trúc dự án

```
Student_Behavior_Recognition_v3/
├── data/                        # Dataset (LOCAL — không có trên GitHub)
│   ├── raw/
│   ├── labeled/
│   └── skeleton/
├── src/
│   ├── common/                  # Dùng chung: log, plot, paths
│   ├── dataset/                 # Loader skeleton
│   ├── preprocess/              # Cắt video, extract skeleton
│   ├── lstm/                    # Thuật toán 1: BiLSTM
│   └── stgcn/                   # Thuật toán 2: ST-GCN
├── models_saved/                # Trọng số model (gitignore)
│   ├── lstm/best.pt
│   └── stgcn/best.pt
├── results/                     # Kết quả train & evaluate
│   ├── lstm/
│   └── stgcn/
├── requirements.txt
└── README.md
```

## Nhãn hành vi (4 lớp)

| Thư mục | Mô tả |
|---------|--------|
| `normal` | Ngồi học bình thường |
| `sleep` | Ngủ gật |
| `standing` | Đứng dậy |
| `use_phone` | Dùng điện thoại |

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quy trình

```text
raw → cut & label → extract skeleton → train → evaluate → plot
```

```powershell
# Tiền xử lý
python src/preprocess/extract_skeleton.py

# --- LSTM ---
python src/lstm/train.py
python src/lstm/evaluate.py
python src/common/plot_history.py --model lstm

# --- ST-GCN ---
python src/stgcn/train.py
python src/stgcn/evaluate.py
python src/common/plot_history.py --model stgcn
```

## Output mỗi model

Mỗi thuật toán có thư mục riêng:

| File | Mô tả |
|------|--------|
| `models_saved/<model>/best.pt` | Trọng số tốt nhất |
| `results/<model>/training_log.csv` | Log từng epoch (loss, acc, P, R, F1) |
| `results/<model>/training_history.png` | Biểu đồ training |
| `results/<model>/test_metrics.json` | Metrics trên test set |
| `results/<model>/confusion_matrix.png` | Ma trận nhầm lẫn |

## Dataset

Dữ liệu không nằm trên GitHub (~50 GB). Xem [`data/README.md`](data/README.md).
