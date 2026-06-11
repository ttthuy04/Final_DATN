"""
Trich xuat skeleton tu video da gan nhan (data/labeled) -> file .npy (data/skeleton).

Moi file .npy co shape (60, 100):
- 60 frame (lay mau deu tu video)
- 100 = 25 landmark (upper body) x 4 (x, y, z, visibility)
"""
import argparse
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
#chuyen skeleton thanhf vector 
N_FRAMES_DEFAULT = 60 
N_LANDMARKS = 25
N_FEATURES = N_LANDMARKS * 4  # 100

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_NAME = "pose_landmarker_lite.task"


def get_model_path(project_root: Path) -> Path:
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_NAME
    if not model_path.exists():
        print(f"Dang tai model MediaPipe: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"Da luu model tai: {model_path}")
    return model_path


def create_pose_landmarker(model_path: Path) -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def landmarks_to_vector(landmarks) -> list[float]:
    vec = []
    for i in range(N_LANDMARKS):
        lm = landmarks[i]
        visibility = getattr(lm, "visibility", None)
        if visibility is None:
            visibility = getattr(lm, "presence", 1.0)
        vec.extend([lm.x, lm.y, lm.z, float(visibility)])
    return vec


def extract_landmarks_from_video(
    video_path: str | Path,
    model_path: Path,
    n_frames: int = N_FRAMES_DEFAULT,
) -> np.ndarray:
    """Moi video dung 1 PoseLandmarker rieng (tranh loi timestamp)."""
    landmarker = create_pose_landmarker(model_path)
    cap = cv2.VideoCapture(str(video_path))
    sequence = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < n_frames:
            skip_frames = 1
        else:
            skip_frames = total_frames // n_frames
        if skip_frames == 0:
            skip_frames = 1

        count = 0
        sample_idx = 0
        ms_step = max(int(1000 / fps), 1)

        while cap.isOpened() and len(sequence) < n_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if count % skip_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = sample_idx * ms_step
                sample_idx += 1
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.pose_landmarks:
                    sequence.append(landmarks_to_vector(results.pose_landmarks[0]))
                elif sequence:
                    sequence.append(sequence[-1])
                else:
                    sequence.append([0.0] * N_FEATURES)

            count += 1
    finally:
        cap.release()
        landmarker.close()

    while len(sequence) < n_frames:
        sequence.append([0.0] * N_FEATURES)

    return np.array(sequence, dtype=np.float32)


def process_all(
    labeled_dir: Path,
    skeleton_dir: Path,
    model_path: Path,
    n_frames: int = N_FRAMES_DEFAULT,
    overwrite: bool = False,
    labels_filter: str = "",
    skip_labels_filter: str = "distracted",
) -> tuple[int, int, int]:
    done = 0
    skipped = 0
    failed = 0

    only_labels = {x.strip() for x in labels_filter.split(",") if x.strip()}
    skip_labels = {x.strip() for x in skip_labels_filter.split(",") if x.strip()}

    labels = sorted(
        d.name for d in labeled_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

    for label in labels:
        if label in skip_labels:
            print(f"\n--- Bo qua nhan: {label} ---")
            continue
        if only_labels and label not in only_labels:
            continue
        label_path = labeled_dir / label
        save_label_path = skeleton_dir / label
        save_label_path.mkdir(parents=True, exist_ok=True)

        videos = sorted(label_path.glob("*.mp4"))
        print(f"\n--- Dang xu ly nhan: {label} ({len(videos)} video) ---")

        for video_path in videos:
            save_name = save_label_path / f"{video_path.stem}.npy"

            if save_name.exists() and not overwrite:
                skipped += 1
                continue

            try:
                skeleton_data = extract_landmarks_from_video(
                    video_path, model_path, n_frames=n_frames
                )
                if skeleton_data.shape != (n_frames, N_FEATURES):
                    raise ValueError(
                        f"Shape sai {skeleton_data.shape}, can ({n_frames}, {N_FEATURES})"
                    )
                np.save(save_name, skeleton_data)
                done += 1
                if done % 50 == 0:
                    print(f"  Da luu {done} file (moi nhat: {save_name.name})")
            except Exception as exc:
                failed += 1
                print(f"  [LOI] {video_path.name}: {exc}")

    return done, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trich xuat skeleton tu data/labeled sang data/skeleton"
    )
    parser.add_argument("--labeled_dir", type=str, default="data/labeled")
    parser.add_argument("--skeleton_dir", type=str, default="data/skeleton")
    parser.add_argument("--n_frames", type=int, default=N_FRAMES_DEFAULT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Chi xu ly cac nhan, cach nhau boi dau phay",
    )
    parser.add_argument(
        "--skip_labels",
        type=str,
        default="distracted",
        help="Bo qua cac nhan (mac dinh: distracted)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    labeled_dir = project_root / args.labeled_dir
    skeleton_dir = project_root / args.skeleton_dir
    model_path = get_model_path(project_root)

    if not labeled_dir.exists():
        raise FileNotFoundError(f"Khong tim thay: {labeled_dir}")

    print("=" * 56)
    print("TRICH XUAT SKELETON (MediaPipe Pose Landmarker)")
    print(f"Input : {labeled_dir}")
    print(f"Output: {skeleton_dir}")
    print(f"Frames/video: {args.n_frames} | Features: {N_FEATURES}")
    print("=" * 56)

    done, skipped, failed = process_all(
        labeled_dir,
        skeleton_dir,
        model_path,
        n_frames=args.n_frames,
        overwrite=args.overwrite,
        labels_filter=args.labels,
        skip_labels_filter=args.skip_labels,
    )

    print("\n" + "=" * 56)
    print("HOAN THANH TRICH XUAT SKELETON")
    print(f"  Moi tao   : {done}")
    print(f"  Bo qua    : {skipped} (da co .npy)")
    print(f"  Loi       : {failed}")
    print("=" * 56)


if __name__ == "__main__":
    main()
