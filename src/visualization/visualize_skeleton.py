"""
Phat lai skeleton tu file .npy (shape 60 x 100) de kiem tra sau extract_skeleton.

Chay:
  python src/visualization/visualize_skeleton.py
  python src/visualization/visualize_skeleton.py --npy data/skeleton/normal/clip_001.npy
  python src/visualization/visualize_skeleton.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.extract_skeleton import N_FEATURES, N_FRAMES_DEFAULT, N_LANDMARKS
from src.stgcn.model import SKELETON_EDGES

DEFAULT_SKELETON_DIR = PROJECT_ROOT / "data" / "skeleton"
SKIP_LABELS = {"distracted"}


def find_sample_npy(skeleton_dir: Path) -> Path | None:
    if not skeleton_dir.is_dir():
        return None
    for label_dir in sorted(skeleton_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name.startswith("."):
            continue
        if label_dir.name in SKIP_LABELS:
            continue
        files = sorted(label_dir.glob("*.npy"))
        if files:
            return files[0]
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize skeleton .npy (60 frames x 25 joints)")
    p.add_argument(
        "--npy",
        type=str,
        default="",
        help="Duong dan file .npy. Bo trong = tu tim mau dau tien trong data/skeleton/",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Liet ke vai file .npy co san roi thoat",
    )
    p.add_argument("--canvas", type=int, default=500, help="Kich thuoc canvas (px)")
    p.add_argument("--fps", type=int, default=20, help="Toc do phat (frame/giay)")
    p.add_argument(
        "--min-vis",
        type=float,
        default=0.3,
        help="Nguong visibility de ve diem/duong",
    )
    return p.parse_args()


def load_skeleton(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay: {path}")
    data = np.load(path)
    if data.ndim == 1 and data.size == N_FEATURES:
        data = data.reshape(1, N_FEATURES)
    if data.ndim != 2 or data.shape[1] != N_FEATURES:
        raise ValueError(
            f"Shape khong hop le: {data.shape}. Can (N, {N_FEATURES}) hoac ({N_FRAMES_DEFAULT}, {N_FEATURES})"
        )
    return data.astype(np.float32)


def frame_to_points(
    frame_data: np.ndarray,
    canvas: int,
    min_vis: float,
) -> list[tuple[int, int, bool]]:
    """Tra ve (x, y, visible) cho 25 khop."""
    points: list[tuple[int, int, bool]] = []
    for j in range(N_LANDMARKS):
        base = j * 4
        x_n, y_n = float(frame_data[base]), float(frame_data[base + 1])
        vis = float(frame_data[base + 3])
        ok = vis >= min_vis and (x_n > 0 or y_n > 0)
        x = int(x_n * canvas)
        y = int(y_n * canvas)
        points.append((x, y, ok))
    return points


def draw_frame(
    canvas: np.ndarray,
    frame_data: np.ndarray,
    canvas_size: int,
    min_vis: float,
) -> None:
    canvas.fill(0)
    points = frame_to_points(frame_data, canvas_size, min_vis)

    for j, (x, y, ok) in enumerate(points):
        if ok:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                str(j),
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 200, 200),
                1,
            )

    for p1, p2 in SKELETON_EDGES:
        if p1 >= len(points) or p2 >= len(points):
            continue
        x1, y1, ok1 = points[p1]
        x2, y2, ok2 = points[p2]
        if ok1 and ok2:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)


def list_samples(skeleton_dir: Path, limit: int = 15) -> None:
    print(f"Thu muc: {skeleton_dir}")
    if not skeleton_dir.is_dir():
        print("  (chua co — hay chay: python src/preprocess/extract_skeleton.py)")
        return
    n = 0
    for label_dir in sorted(skeleton_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        files = sorted(label_dir.glob("*.npy"))
        if not files:
            continue
        print(f"\n  [{label_dir.name}] ({len(files)} file)")
        for f in files[:3]:
            print(f"    {f.relative_to(PROJECT_ROOT)}")
        n += len(files)
    print(f"\nTong: {n} file .npy")


def main() -> None:
    args = parse_args()
    skeleton_dir = DEFAULT_SKELETON_DIR

    if args.list:
        list_samples(skeleton_dir)
        return

    if args.npy:
        npy_path = Path(args.npy)
        if not npy_path.is_absolute():
            npy_path = PROJECT_ROOT / npy_path
    else:
        npy_path = find_sample_npy(skeleton_dir)
        if npy_path is None:
            print("Khong tim thay file .npy trong data/skeleton/.")
            print("Hay chay extract_skeleton truoc, hoac chi ro duong dan:")
            print("  python src/visualization/visualize_skeleton.py --npy data/skeleton/normal/xxx.npy")
            print("  python src/visualization/visualize_skeleton.py --list")
            sys.exit(1)
        print(f"Tu dong chon: {npy_path.relative_to(PROJECT_ROOT)}")

    data = load_skeleton(npy_path)
    canvas_size = args.canvas
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    delay_ms = max(1, int(1000 / args.fps))

    win = f"Skeleton — {npy_path.name} ({data.shape[0]} frames)"
    print(win)
    print("Nhan 'q' hoac ESC de thoat.")

    for i, frame_data in enumerate(data):
        draw_frame(canvas, frame_data, canvas_size, args.min_vis)
        cv2.putText(
            canvas,
            f"frame {i + 1}/{len(data)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.imshow(win, canvas)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
