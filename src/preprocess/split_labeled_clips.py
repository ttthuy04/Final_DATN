"""
Cat cac clip trong data/labeled qua dai thanh nhieu doan ngan (mac dinh 5 giay).
Muc tieu: tang so mau train, phu hop extract skeleton (60 frame / clip ngan).

Video <= max_sec giu nguyen. Video dai hon -> tach thanh nhieu file, xoa file goc.
"""
import argparse
from pathlib import Path

import cv2


def split_video(
    video_path: Path,
    max_sec: float,
    min_tail_sec: float,
) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0

    if duration <= max_sec:
        cap.release()
        return []

    frames_per_seg = int(max_sec * fps)
    min_tail_frames = int(min_tail_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    created = []
    seg_idx = 1
    start = 0

    while start < total_frames:
        end = min(start + frames_per_seg, total_frames)
        seg_len = end - start
        if seg_len < min_tail_frames and created:
            break
        if seg_len < min_tail_frames and not created:
            break

        out_path = video_path.with_name(f"{video_path.stem}_seg{seg_idx:03d}.mp4")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(seg_len):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
        created.append(out_path)
        seg_idx += 1
        start = end

    cap.release()
    if created:
        try:
            video_path.unlink()
        except OSError as exc:
            print(f"  [WARN] Khong xoa file goc {video_path.name}: {exc}")
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Tach clip labeled qua dai")
    parser.add_argument("--labeled_dir", type=str, default="data/labeled")
    parser.add_argument("--max_sec", type=float, default=5.0, help="Do dai toi da moi clip")
    parser.add_argument("--min_tail_sec", type=float, default=2.0, help="Bo doan cuoi neu ngan hon")
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Chi xu ly cac nhan, cach nhau boi dau phay (vd: normal,sleep). Mac dinh: tat ca",
    )
    parser.add_argument(
        "--skip_labels",
        type=str,
        default="distracted",
        help="Bo qua cac nhan (mac dinh: distracted)",
    )
    args = parser.parse_args()

    labeled_dir = Path(__file__).resolve().parents[2] / args.labeled_dir
    if not labeled_dir.exists():
        raise FileNotFoundError(labeled_dir)

    only_labels = {x.strip() for x in args.labels.split(",") if x.strip()}
    skip_labels = {x.strip() for x in args.skip_labels.split(",") if x.strip()}

    split_count = 0
    kept = 0

    for label_dir in sorted(labeled_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        if label_dir.name in skip_labels:
            print(f"\n--- Bo qua nhan: {label_dir.name} ---")
            continue
        if only_labels and label_dir.name not in only_labels:
            continue
        videos = sorted(label_dir.glob("*.mp4"))
        print(f"\n--- {label_dir.name}: {len(videos)} file ---")
        for video_path in videos:
            if "_seg" in video_path.stem:
                continue
            try:
                created = split_video(video_path, args.max_sec, args.min_tail_sec)
            except Exception as exc:
                print(f"  [LOI] Bo qua {video_path.name}: {exc}")
                continue
            if created:
                split_count += len(created)
                print(f"  {video_path.name} -> {len(created)} clip")
            else:
                kept += 1

    print(f"\nTong clip moi tao: {split_count} | Giu nguyen: {kept}")
    print("Chay tiep: python src/preprocess/rename_labeled.py")
    print("           python src/preprocess/extract_skeleton.py")


if __name__ == "__main__":
    main()
