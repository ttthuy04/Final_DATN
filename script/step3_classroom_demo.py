"""
Bước 3 — Demo cuối: YOLO track -> skeleton -> ST-GCN -> nhãn + thống kê lớp.

Chạy:
  python script/step3_classroom_demo.py
  python script/step3_classroom_demo.py --video data/raw/test1.mp4

Output:
  results/demo_check/classroom_demo_final.mp4
  results/demo_check/classroom_demo_log.csv
"""
import argparse
import csv
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import ModelPaths
from src.preprocess.extract_skeleton import N_FEATURES, N_FRAMES_DEFAULT, get_model_path, landmarks_to_vector
from src.stgcn.model import SkeletonSTGCN

DEFAULT_VIDEO = PROJECT_ROOT / "data" / "raw" / "test1.mp4"
DEFAULT_OUT = PROJECT_ROOT / "results" / "demo_check" / "classroom_demo_final.mp4"
DEFAULT_LOG = PROJECT_ROOT / "results" / "demo_check" / "classroom_demo_log.csv"

PERSON_CLASS = 0
WINDOW = N_FRAMES_DEFAULT

LABEL_VI = {
    "normal": "Binh thuong",
    "sleep": "Ngu gat",
    "standing": "Dung day",
    "use_phone": "Dung DT",
}

LABEL_COLOR = {
    "normal": (0, 200, 0),
    "sleep": (0, 0, 255),
    "standing": (255, 160, 0),
    "use_phone": (0, 220, 255),
}


@dataclass
class TrackState:
    """buffer: chi 60 frame skeleton MediaPipe thanh cong."""
    buffer: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    label: str = ""  # chua du du lieu -> khong hien nhan
    conf: float = 0.0
    recent_preds: deque = field(default_factory=lambda: deque(maxlen=7))


def parse_args():
    p = argparse.ArgumentParser(description="Demo nhan dien hanh vi lop hoc (ST-GCN)")
    p.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    p.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--log-csv", type=str, default=str(DEFAULT_LOG))
    p.add_argument("--yolo-model", type=str, default="yolo11s.pt")
    p.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Nguong YOLO person; tang neu nhan nham ban/ghe",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--padding", type=float, default=0.15)
    p.add_argument("--min-box-h", type=int, default=60)
    p.add_argument(
        "--predict-every",
        type=int,
        default=60,
        help="ST-GCN moi N frame (~2s @ 30fps). Lon hon = cap nhat cham, on dinh hon",
    )
    p.add_argument(
        "--smooth-votes",
        type=int,
        default=7,
        help="So lan du doan gan nhat de vote nhan (can dong thuan)",
    )
    p.add_argument(
        "--min-agree",
        type=int,
        default=5,
        help="It nhat N/7 lan du doan trung nhan moi doi nhan hien thi",
    )
    p.add_argument(
        "--min-conf",
        type=float,
        default=0.55,
        help="Bo qua du doan co confidence thap hon nguong nay",
    )
    p.add_argument("--max-frames", type=int, default=0, help="0 = het video")
    return p.parse_args()


def create_image_landmarker(model_path: Path) -> PoseLandmarker:
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(opts)


def load_stgcn(device):
    paths = ModelPaths("stgcn")
    ckpt_path = paths.model_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Chua co ST-GCN: {ckpt_path}. Chay: python src/stgcn/train.py")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    names = ckpt["class_names"]
    model = SkeletonSTGCN(
        num_classes=len(names),
        dropout=ckpt.get("dropout", 0.25),
        use_velocity=ckpt.get("use_velocity", True),
        normalize_root=ckpt.get("normalize_root", True),
        use_spatial_partition=ckpt.get("use_spatial_partition", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, names


@torch.no_grad()
def predict_behavior(model, class_names, seq: np.ndarray, device) -> tuple[str, float, np.ndarray]:
    x = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32, device=device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx]), probs


def boxes_from_results(results) -> list[tuple[int, int, int, int, int]]:
    out = []
    for box in results.boxes:
        if int(box.cls[0]) != PERSON_CLASS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tid = int(box.id[0]) if box.id is not None else -1
        out.append((tid, x1, y1, x2, y2))
    return out


def expand_bbox(x1, y1, x2, y2, w, h, pad: float):
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0, int(x1 - pad * bw)),
        max(0, int(y1 - pad * bh)),
        min(w, int(x2 + pad * bw)),
        min(h, int(y2 + pad * bh)),
    )


def skeleton_from_crop(landmarker, crop_bgr) -> np.ndarray | None:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_img)
    if not res.pose_landmarks:
        return None
    return np.array(landmarks_to_vector(res.pose_landmarks[0]), dtype=np.float32)


def stable_label(track: TrackState, min_agree: int) -> str | None:
    """Tra ve nhan moi neu du dong thuan; None = giu nhan cu."""
    if len(track.recent_preds) < min_agree:
        return None
    counts: dict[str, int] = defaultdict(int)
    for lbl in track.recent_preds:
        counts[lbl] += 1
    best, cnt = max(counts.items(), key=lambda x: x[1])
    if cnt >= min_agree:
        return best
    return None


def draw_stats_panel(frame, counts: dict[str, int], class_names: list[str]):
    x0, y0 = frame.shape[1] - 320, 10
    cv2.rectangle(frame, (x0, y0), (frame.shape[1] - 10, y0 + 30 + 28 * (len(class_names) + 1)), (30, 30, 30), -1)
    cv2.putText(frame, "THONG KE LOP", (x0 + 10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    total = sum(counts.values())
    cv2.putText(frame, f"Tong SV: {total}", (x0 + 10, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    for i, cls in enumerate(class_names):
        n = counts.get(cls, 0)
        vi = LABEL_VI.get(cls, cls)
        col = LABEL_COLOR.get(cls, (255, 255, 255))
        cv2.putText(frame, f"{vi}: {n}", (x0 + 10, y0 + 72 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)


def main():
    args = parse_args()
    video = Path(args.video)
    out_path = Path(args.output)
    log_path = Path(args.log_csv)

    if not video.exists():
        raise FileNotFoundError(f"Khong thay video: {video}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Buoc 3: Demo cuoi (YOLO + Skeleton + ST-GCN) ===")
    print(f"Video  : {video}")
    print(f"Output : {out_path}")
    print(f"Device : {device}")
    print(
        f"Cap nhat nhan: moi {args.predict_every} frame | "
        f"vote {args.min_agree}/{args.smooth_votes} | min_conf={args.min_conf}"
    )

    stgcn, class_names = load_stgcn(device)
    yolo = YOLO(args.yolo_model)
    landmarker = create_image_landmarker(get_model_path(PROJECT_ROOT))

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_frames > 0:
        total = min(total, args.max_frames)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tracks: dict[int, TrackState] = {}
    log_rows = []
    frame_idx = 0
    t0 = time.time()

    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            classes=[PERSON_CLASS],
            imgsz=args.imgsz,
            iou=0.45,
            device="cpu",
            verbose=False,
        )[0]
        boxes = boxes_from_results(results)
        frame_vis = frame.copy()
        active_ids = set()
        frame_counts: dict[str, int] = defaultdict(int)

        for pi, (tid, x1, y1, x2, y2) in enumerate(boxes):
            if tid < 0:
                tid = 10000 + pi
            if (y2 - y1) < args.min_box_h:
                continue
            ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, w, h, args.padding)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue

            active_ids.add(tid)
            if tid not in tracks:
                tracks[tid] = TrackState()

            tr = tracks[tid]
            if len(tr.recent_preds) == 0:
                tr.recent_preds = deque(maxlen=args.smooth_votes)

            vec = skeleton_from_crop(landmarker, crop)
            if vec is not None:
                tr.buffer.append(vec)

            if len(tr.buffer) == WINDOW and frame_idx % args.predict_every == 0:
                seq = np.stack(list(tr.buffer), axis=0)
                lbl, conf, _ = predict_behavior(stgcn, class_names, seq, device)
                if conf >= args.min_conf:
                    tr.recent_preds.append(lbl)
                    new_lbl = stable_label(tr, args.min_agree)
                    if new_lbl is not None:
                        tr.label = new_lbl
                        tr.conf = conf
                    log_rows.append(
                        {
                            "frame": frame_idx,
                            "track_id": tid,
                            "label": tr.label or lbl,
                            "conf": round(conf, 4),
                            "raw_pred": lbl,
                        }
                    )

            if tr.label:
                lbl_vi = LABEL_VI.get(tr.label, tr.label)
                color = LABEL_COLOR.get(tr.label, (0, 255, 0))
                frame_counts[tr.label] += 1
            else:
                lbl_vi = "..."
                color = (180, 180, 180)

            cv2.rectangle(frame_vis, (ex1, ey1), (ex2, ey2), color, 2)
            conf_txt = f"{tr.conf * 100:.0f}%" if tr.label else "--"
            txt = f"ID{tid} {lbl_vi} {conf_txt}"
            cv2.putText(
                frame_vis,
                txt,
                (ex1, max(ey1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        draw_stats_panel(frame_vis, frame_counts, class_names)
        cv2.putText(
            frame_vis,
            f"Frame {frame_idx}/{total}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(frame_vis)
        frame_idx += 1

        if frame_idx % 30 == 0:
            pct = frame_idx / total * 100
            print(f"  [{frame_idx}/{total}] {pct:.0f}%", flush=True)

    cap.release()
    landmarker.close()
    writer.release()

    log_fields = ["frame", "track_id", "label", "conf", "raw_pred"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=log_fields, extrasaction="ignore")
        wcsv.writeheader()
        wcsv.writerows(log_rows)

    elapsed = time.time() - t0
    print(f"\nHoan thanh trong {elapsed:.0f}s")
    print(f"Video : {out_path}")
    print(f"Log   : {log_path} ({len(log_rows)} dong du doan)")
    print(f"Tracks: {len(tracks)}")


if __name__ == "__main__":
    main()
