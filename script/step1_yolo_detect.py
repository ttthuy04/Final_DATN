"""
Bước 1: YOLO detect person trên video lớp — tối ưu cho máy cá nhân (CPU).

Model khuyến nghị (YOLO11, nhanh -> chậm):
  yolo11n.pt   nhanh nhat (CPU)
  yolo11s.pt   can bang, chinh xac hon (mac dinh)
  yolo11m/l/x  qua nang cho CPU

Chạy nhanh (mặc định):
  python script/step1_yolo_detect.py

Chạy đầy đủ mọi frame:
  python script/step1_yolo_detect.py --detect-every 1

Test 300 frame đầu:
  python script/step1_yolo_detect.py --max-frames 300
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "raw" / "test1.mp4"
DEFAULT_OUT = PROJECT_ROOT / "results" / "demo_check" / "step1_yolo_output.mp4"

PERSON_CLASS = 0


def parse_args():
    p = argparse.ArgumentParser(description="YOLO detect person — tối ưu CPU")
    p.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    p.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    p.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="yolo11n | yolo11s (mac dinh) | yolo11m",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Cao hon = it box rac (ban/ghe). Thap hon (0.15-0.2) neu miss nguoi guc ban",
    )
    p.add_argument("--imgsz", type=int, default=640, help="640 nhanh, 960/1280 cham hon")
    p.add_argument(
        "--detect-every",
        type=int,
        default=2,
        help="Chi chay YOLO moi N frame, frame giua tai su dung box cu (nhanh hon)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Gioi han so frame (0 = het video)",
    )
    return p.parse_args()


def boxes_from_results(results) -> list[tuple[int, int, int, int]]:
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) == PERSON_CLASS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes


def draw_boxes(frame, boxes, label: str):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
    )


def main():
    args = parse_args()
    video = Path(args.video)
    out_path = Path(args.output)

    if not video.exists():
        raise FileNotFoundError(f"Không thấy video: {video}")

    print(f"Video  : {video}")
    print(f"Model  : {args.model} | conf={args.conf} | imgsz={args.imgsz}")
    print(f"Tối ưu : detect mỗi {args.detect_every} frame (CPU-friendly)")
    if args.max_frames:
        print(f"Giới hạn: {args.max_frames} frame")

    t0 = time.time()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    last_boxes: list[tuple[int, int, int, int]] = []
    counts = []
    frame_idx = 0

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        run_detect = (frame_idx % args.detect_every == 0) or not last_boxes
        if run_detect:
            results = model(
                frame,
                conf=args.conf,
                classes=[PERSON_CLASS],
                imgsz=args.imgsz,
                iou=0.45,
                device="cpu",
                verbose=False,
            )[0]
            last_boxes = boxes_from_results(results)

        n_person = len(last_boxes)
        counts.append(n_person)
        label = (
            f"Persons: {n_person} | {args.model} | conf={args.conf} | "
            f"detect/ {args.detect_every}f"
        )
        draw_boxes(frame, last_boxes, label)
        writer.write(frame)
        frame_idx += 1

        if frame_idx % 60 == 0:
            elapsed = time.time() - t0
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / fps_proc if fps_proc > 0 else 0
            pct = frame_idx / total_frames * 100
            print(
                f"  [{frame_idx}/{total_frames}] {pct:.0f}% | "
                f"{fps_proc:.1f} fps | ETA ~{eta:.0f}s | nguoi: {n_person}",
                flush=True,
            )

    cap.release()
    writer.release()
    elapsed = time.time() - t0

    avg = sum(counts) / len(counts) if counts else 0
    print(f"\nHoàn thành trong {elapsed:.0f}s ({frame_idx / elapsed:.1f} fps xử lý)")
    print(f"Frames: {frame_idx}")
    print(f"Trung bình người/frame: {avg:.1f}")
    print(f"Min: {min(counts) if counts else 0} | Max: {max(counts) if counts else 0}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
