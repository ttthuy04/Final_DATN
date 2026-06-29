# Student Behavior Recognition

A deep learning system for recognizing student behaviors in classroom videos using skeleton-based action recognition.

---

## Features

- Human detection with YOLOv8
- Skeleton extraction using MediaPipe Pose (25 upper-body landmarks)
- Behavior recognition using **ST-GCN** (main model) and **LSTM** (baseline)
- Web-based demo with Flask (upload video → visualize result)

---

## Tech Stack

- Python 3.10+
- PyTorch
- OpenCV
- MediaPipe
- YOLOv8 (Ultralytics)
- scikit-learn
- Flask

---

## Requirements

- Python 3.10 or 3.11 (recommended)
- Git
- (Optional) NVIDIA GPU with CUDA — speeds up training 10–20×

---

## Installation

### Clone repository

```bash
git clone <repository-url>
cd Student_Behavior_Recognition_v3
```

### Create virtual environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset

```
data/
├── raw/         # Original classroom videos
├── labeled/     # Clips cut and labeled by behavior
│   ├── normal/
│   ├── sleep/
│   ├── standing/
│   └── use_phone/
└── skeleton/    # Extracted skeleton .npy files
```

Classes

- Normal
- Sleep
- Standing
- Use Phone

Skeleton format

- `.npy` shape per clip: `(60, 100)` = 60 frames × (25 joints × 4 values: `x, y, z, visibility`)

Model input

- LSTM : `(N, 60, 100)`
- ST-GCN : `(N, 4, 60, 25)` after `flat_to_graph()` (channels = x, y + velocity)

## Pipeline

```
Raw video → Cut & label → Extract skeleton → Train → Evaluate → Demo / Web
```

### Prepare data

```bash
# Cut and label clips (interactive)
python src/preprocess/cut_video.py --video data/raw/video_demo.mp4

# Extract skeleton (.npy)
python src/preprocess/extract_skeleton.py
```

---

## Training

ST-GCN (main model)

```bash
python src/stgcn/train.py
```

LSTM (baseline)

```bash
python src/lstm/train.py
```

Both models use a shared training loop at `src/common/train_loop.py` with the same configuration (Adam + ReduceLROnPlateau, gradient clipping, early stopping on F1 macro) for a fair comparison.

Outputs

- `models_saved/<model>/best.pt`
- `results/<model>/training_log.csv`

---

## Evaluation

```bash
python src/stgcn/evaluate.py
python src/lstm/evaluate.py
```

Reports Accuracy, Precision, Recall, F1 (macro), per-class classification report, and a confusion matrix image.

Outputs

- `results/<model>/test_metrics.json`
- `results/<model>/confusion_matrix.png`

Plot training history

```bash
python src/common/plot_history.py --model stgcn
```

---

## Demo

End-to-end classroom demo (YOLOv8 → MediaPipe → ST-GCN):

```bash
python script/step3_classroom_demo.py --video data/raw/test1.mp4
```

Outputs a result video and `result.json` under `results/demo_check/<video_name>_<timestamp>/`.

Step-by-step scripts (for debugging the pipeline):

```bash
python script/step1_yolo_detect.py     --video data/raw/test1.mp4
python script/step2_mediapipe_crop.py  --video data/raw/test1.mp4
```

---

## Web Demo

```bash
python web/server.py
```

Open

```
http://127.0.0.1:5050
```

Upload a video → server runs the full pipeline → shows the annotated video and per-track statistics.

---

## Project Structure

```
Student_Behavior_Recognition_v3/
├── data/             # Local dataset (gitignored)
├── models/           # MediaPipe model (auto-downloaded)
├── models_saved/     # Trained weights (gitignored)
├── results/          # Logs, metrics, confusion matrices, demo runs
├── src/
│   ├── common/       # Paths, shared train loop, log, plot, YOLO filter
│   ├── dataset/      # Skeleton loader, train/val/test split
│   ├── demo/         # Classroom demo pipeline
│   ├── lstm/         # BiLSTM: model, train, evaluate
│   ├── stgcn/        # ST-GCN: model, train, evaluate
│   ├── preprocess/   # Cut video, extract skeleton
│   └── visualization/# Draw skeleton on frames
├── script/           # Step-by-step demo scripts
├── web/              # Flask backend + frontend
├── requirements.txt
└── README.md
```
