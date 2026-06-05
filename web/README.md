# Web demo — bao ve do an

Giao dien don gian: **Upload video → Chay demo → Video ket qua + Bang thong ke**.

## Yeu cau

- Da train ST-GCN: `models_saved/stgcn/best.pt`
- Da cai dependencies: `pip install -r requirements.txt`

## Cau hinh mac dinh (chinh xac)

Tat ca script dung **mot preset** trong `src/common/demo_presets.py`:

| Tang | Cong nghe | Muc tieu |
|------|-----------|----------|
| Detect | YOLOv8m, conf=0.10, imgsz=1280 | Bat nguoi (ke ca guc ban), loc chan ban |
| Skeleton | MediaPipe + pose gate | Chi giu box co khung xuong hop le |
| Hanh vi | ST-GCN, vote 3/5, min_conf=0.55 | Nhan on dinh, it nhãn sai |

## Chay web

```powershell
cd c:\Documen\A_DATN\Student_Behavior_Recognition_v3
python web/app.py
```

Mo trinh duyet: **http://127.0.0.1:7860**

## Luong su dung

1. Upload video lop hoc (`.mp4`, toi da ~200 MB)
2. Bam **Chay demo** (hoac **Dung video mau** de lay `data/raw/test1.mp4`)
3. Doi xu ly (CPU co the 5–10 phut voi video dai)
4. Xem video co box + nhan hanh vi va bang thong ke 4 lop

## Output luu local

| Thu muc | Noi dung |
|---------|----------|
| `web/uploads/` | Video upload (ban sao) |
| `web/outputs/` | Video ket qua, log CSV, stats JSON |

## CLI tuong duong

```powershell
python script/step3_classroom_demo.py --video data/raw/test11.mp4
```

## Cau truc code

```
web/
├── app.py           # Gradio UI
└── demo_runner.py   # Upload + goi pipeline

src/demo/
└── classroom_demo.py  # Pipeline YOLO + Skeleton + ST-GCN (dung chung CLI & web)
```
