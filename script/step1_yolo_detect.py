"""

Bước 1: YOLO detect person trên video lớp — tối ưu cho máy cá nhân (CPU).



Model khuyến nghị (nhanh -> cham, chinh xac):

  yolov8n.pt   nhanh nhat (CPU)

  yolov8m.pt   can bang, detect tot hon (mac dinh)

  yolov8x.pt   chinh xac nhat nhung rat cham tren CPU



Loc bbox thong minh: bo chan ban/ghe, giu nguoi guc ban.



Chạy nhanh (mặc định):

  python script/step1_yolo_detect.py



Chạy đầy đủ mọi frame:

  python script/step1_yolo_detect.py --detect-every 1

"""

import argparse

import sys

import time

from pathlib import Path



import cv2

from ultralytics import YOLO



PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))



from src.common.demo_presets import ACCURACY_YOLO_MODEL, accuracy_yolo_config
from src.common.yolo_person_filter import (
    extract_person_boxes,
    filter_person_boxes,
    yolo_predict_kwargs,
)



DEFAULT_VIDEO = PROJECT_ROOT / "data" / "raw" / "test1.mp4"

DEFAULT_OUT = PROJECT_ROOT / "results" / "demo_check" / "step1_yolo_output.mp4"





def parse_args():

    p = argparse.ArgumentParser(description="YOLO detect person — loc bbox lop hoc")

    p.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))

    p.add_argument("--output", type=str, default=str(DEFAULT_OUT))

    p.add_argument(

        "--model",

        type=str,

        default=ACCURACY_YOLO_MODEL,

        help="yolov8n | yolov8m (mac dinh) | yolov8x | yolo11s",

    )

    p.add_argument(

        "--conf",

        type=float,

        default=0.10,

        help="Thap hon = bat nguoi guc ban. Loc hinh hoc loai chan ban",

    )

    p.add_argument("--imgsz", type=int, default=1280)

    p.add_argument("--min-box-h", type=int, default=25)

    p.add_argument(

        "--detect-every",

        type=int,

        default=1,

        help="Chi chay YOLO moi N frame (1 = chinh xac nhat)",

    )

    p.add_argument("--max-frames", type=int, default=0)

    return p.parse_args()





def draw_boxes(frame, boxes, label: str):

    for box in boxes:

        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)

        cv2.putText(

            frame,

            f"{box.conf * 100:.0f}%",

            (box.x1, max(box.y1 - 6, 14)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (0, 255, 0),

            1,

        )

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



    yolo_cfg = accuracy_yolo_config()
    yolo_kw = yolo_predict_kwargs(yolo_cfg)

    print(f"Video  : {video}")
    print(f"Model  : {args.model} | preset CHINH XAC | conf={yolo_cfg.conf} | imgsz={yolo_cfg.imgsz}")

    print(f"Detect : moi {args.detect_every} frame")



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

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))



    last_boxes = []

    counts = []

    frame_idx = 0



    while frame_idx < total_frames:

        ret, frame = cap.read()

        if not ret:

            break



        run_detect = (frame_idx % args.detect_every == 0) or not last_boxes

        if run_detect:

            results = model(frame, **yolo_kw)[0]

            raw = extract_person_boxes(results, h, w)

            last_boxes = filter_person_boxes(raw, h, w, yolo_cfg, use_pose_gate=False)



        n_person = len(last_boxes)

        counts.append(n_person)

        label = f"Persons: {n_person} | {args.model} | conf={args.conf}"

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


