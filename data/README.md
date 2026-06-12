# Cấu trúc thư mục `data/`

Thư mục này **không được đẩy lên GitHub** (video và nhãn chiếm ~50 GB).
Mỗi máy clone repo cần **tự đặt dữ liệu** vào đúng cấu trúc bên dưới.

## Cây thư mục

```
data/
├── raw/          # Video gốc (.mp4) từ camera/lớp học
├── labeled/      # Video đã cắt và gán nhãn hành vi
│   ├── normal/
│   ├── sleep/
│   ├── standing/
│   └── use_phone/
└── skeleton/     # Keypoint/skeleton trích xuất (đầu ra preprocess)
```

## Các bước chuẩn bị dữ liệu (local)

1. Đặt video gốc vào `data/raw/`.
2. Chạy tiền xử lý (cắt clip, gán nhãn) → lưu vào `data/labeled/<tên_lớp>/`.
3. Chạy `src/preprocess/extract_skeleton.py` → lưu skeleton vào `data/skeleton/`.

## Lưu trữ & sao lưu

- Sao lưu `data/` bằng ổ cứng ngoài, NAS, OneDrive/Google Drive, hoặc bản sao trên máy khác.
- Trong báo cáo/đồ án: ghi rõ nguồn dataset và đường dẫn lưu trữ ngoài repo.

