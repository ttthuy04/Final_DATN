"""
Bước 2: YOLO bbox -> crop từng người -> MediaPipe skeleton.

Mục tiêu: kiểm tra skeleton trên crop có đủ tốt cho ST-GCN (60 frame x 100).

Chạy nhanh (90 frame đầu):
  python script/step2_mediapipe_crop.py

Full video:
  python script/step2_mediapipe_crop.py --max-frames 0

Video khác:
  python script/step2_mediapipe_crop.py --video data/raw/test1.mp4
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.extract_skeleton import N_FEATURES, N_LANDMARKS, get_model_path, landmarks_to_vector

DEFAULT_VIDEO = PROJECT_ROOT / "data" / "raw" / "test1.mp4"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "demo_check" / "step2_crops"
DEFAULT_PREVIEW = PROJECT_ROOT / "results" / "demo_check" / "step2_skeleton_preview.mp4"

PERSON_CLASS = 0
SKELETON_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
]


def parse_args():
    p = argparse.ArgumentParser(description="Bước 2: skeleton trên crop từ YOLO bbox")
    p.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--preview", type=str, default=str(DEFAULT_PREVIEW))
    p.add_argument("--model", type=str, default="yolo11s.pt")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="MediaPipe moi N frame / track (YOLO track van chay moi frame)",
    )
    p.add_argument(
        "--no-track",
        action="store_true",
        help="Tat ByteTrack — ID nguoi co the nhay giua cac frame",
    )
    p.add_argument("--padding", type=float, default=0.15, help="Mo rong bbox (ty le)")
    p.add_argument("--min-box-h", type=int, default=60, help="Bo qua bbox qua thap")
    p.add_argument("--max-frames", type=int, default=90, help="0 = het video")
    p.add_argument(
        "--save-crop-images",
        type=int,
        default=40,
        help="So anh crop mau luu ra (0 = khong luu)",
    )
    p.add_argument("--no-preview", action="store_true", help="Khong ghi video preview")
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


def boxes_from_yolo(results) -> list[tuple[int, int, int, int, int]]:
    """(track_id, x1, y1, x2, y2). track_id=-1 neu khong tracking."""
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) != PERSON_CLASS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tid = -1
        if box.id is not None:
            tid = int(box.id[0])
        boxes.append((tid, x1, y1, x2, y2))
    return boxes


def run_yolo(frame, yolo, args, use_track: bool):
    kwargs = dict(
        conf=args.conf,
        classes=[PERSON_CLASS],
        imgsz=args.imgsz,
        iou=0.45,
        device="cpu",
        verbose=False,
    )
    if use_track:
        return yolo.track(frame, persist=True, tracker="bytetrack.yaml", **kwargs)[0]
    return yolo(frame, **kwargs)[0]


def expand_bbox(x1, y1, x2, y2, w, h, pad: float):
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw))
    y2 = min(h, int(y2 + pad * bh))
    return x1, y1, x2, y2


def draw_landmarks_on_crop(crop_bgr: np.ndarray, lms) -> np.ndarray:
    vis = crop_bgr.copy()
    ch, cw = vis.shape[:2]
    pts = []
    for i in range(min(N_LANDMARKS, len(lms))):
        cx = int(lms[i].x * cw)
        cy = int(lms[i].y * ch)
        pts.append((cx, cy))
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
    for a, b in SKELETON_EDGES:
        if a < len(pts) and b < len(pts):
            cv2.line(vis, pts[a], pts[b], (255, 120, 0), 2)
    cv2.putText(vis, "OK", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


def draw_landmarks_on_frame(frame, x1, y1, crop_w, crop_h, lms) -> None:
    pts = []
    for i in range(min(N_LANDMARKS, len(lms))):
        px = int(x1 + lms[i].x * crop_w)
        py = int(y1 + lms[i].y * crop_h)
        pts.append((px, py))
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
    for a, b in SKELETON_EDGES:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 120, 0), 2)


def detect_skeleton_on_crop(landmarker: PoseLandmarker, crop_bgr: np.ndarray):
    """Tra ve (vector 100 hoac None, landmarks, vis_crop)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_img)
    if not res.pose_landmarks:
        vis = crop_bgr.copy()
        cv2.putText(vis, "FAIL", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, None, vis

    lms = res.pose_landmarks[0]
    vec = landmarks_to_vector(lms)
    vis = draw_landmarks_on_crop(crop_bgr, lms)
    return vec, lms, vis


def main():
    args = parse_args()
    video = Path(args.video)
    out_dir = Path(args.out_dir)
    preview_path = Path(args.preview)

    if not video.exists():
        raise FileNotFoundError(f"Khong thay video: {video}")

    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Buoc 2: YOLO crop + MediaPipe skeleton ===")
    print(f"Video : {video}")
    print(f"Crops : {out_dir}")
    if not args.no_preview:
        print(f"Preview: {preview_path}")

    mp_path = get_model_path(PROJECT_ROOT)
    yolo = YOLO(args.model)
    landmarker = create_image_landmarker(mp_path)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_frames > 0:
        total = min(total, args.max_frames)

    writer = None
    if not args.no_preview:
        writer = cv2.VideoWriter(
            str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    use_track = not args.no_track
    last_boxes: list[tuple[int, int, int, int, int]] = []
    ok_count = 0
    fail_count = 0
    saved_images = 0
    frame_idx = 0
    t0 = time.time()

    print(f"Tracking: {'ByteTrack (on)' if use_track else 'OFF — ID co the nhay'}")

    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO track moi frame de giu ID on dinh; skeleton co the moi N frame
        results = run_yolo(frame, yolo, args, use_track)
        last_boxes = boxes_from_yolo(results)

        frame_vis = frame.copy()
        n_ok_frame = 0
        run_skel = (frame_idx % args.detect_every == 0)

        for pi, (track_id, x1, y1, x2, y2) in enumerate(last_boxes):
            if (y2 - y1) < args.min_box_h:
                continue
            ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, w, h, args.padding)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue

            id_label = f"ID{track_id}" if track_id >= 0 else f"P{pi}"
            status = "SKIP"
            crop_vis = crop.copy()
            lms = None

            if run_skel:
                vec, lms, crop_vis = detect_skeleton_on_crop(landmarker, crop)
                status = "OK" if vec is not None else "FAIL"
                if vec is not None:
                    ok_count += 1
                    n_ok_frame += 1
                    crop_h, crop_w = crop.shape[:2]
                    draw_landmarks_on_frame(frame_vis, ex1, ey1, crop_w, crop_h, lms)
                else:
                    fail_count += 1

            cv2.rectangle(frame_vis, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)
            cv2.putText(
                frame_vis,
                f"{id_label}:{status}",
                (ex1, max(ey1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if status == "OK" else (0, 0, 255) if status == "FAIL" else (0, 255, 255),
                2,
            )

            if (
                run_skel
                and args.save_crop_images > 0
                and saved_images < args.save_crop_images
            ):
                name = f"f{frame_idx:04d}_{id_label}_{status}.jpg"
                cv2.imwrite(str(out_dir / name), crop_vis)
                saved_images += 1

        label = f"Frame {frame_idx} | boxes={len(last_boxes)} | skel OK={n_ok_frame}"
        cv2.putText(
            frame_vis, label, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        if writer:
            writer.write(frame_vis)

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx / total * 100
            rate = ok_count / (ok_count + fail_count) * 100 if (ok_count + fail_count) else 0
            print(
                f"  [{frame_idx}/{total}] {pct:.0f}% | skeleton OK: {rate:.1f}%",
                flush=True,
            )

    cap.release()
    landmarker.close()
    if writer:
        writer.release()

    total_tries = ok_count + fail_count
    rate = ok_count / total_tries * 100 if total_tries else 0
    elapsed = time.time() - t0

    print(f"\nHoan thanh trong {elapsed:.0f}s")
    print(f"Crop da xu ly     : {total_tries}")
    print(f"Skeleton OK       : {ok_count} ({rate:.1f}%)")
    print(f"Skeleton FAIL     : {fail_count}")
    print(f"Anh mau luu       : {saved_images} -> {out_dir}")
    if not args.no_preview:
        print(f"Video preview     : {preview_path}")
    print(f"Vector moi lan OK : shape ({N_FEATURES},) — gom 60 frame o buoc 3 (ST-GCN)")


if __name__ == "__main__":
    main()
