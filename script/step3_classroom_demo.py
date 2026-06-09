"""
Demo chinh xac: YOLOv8m -> skeleton (MediaPipe) -> ST-GCN -> video + thong ke.

Cau hinh mac dinh: src/common/demo_presets.py (uu tien chinh xac).

Chay:
  python script/step3_classroom_demo.py --video data/raw/test11.mp4
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.demo_check_paths import make_demo_check_run_dir, step3_output_path
from src.common.demo_presets import ACCURACY_YOLO_MODEL, accuracy_yolo_config
from src.demo.classroom_demo import DemoConfig, run_classroom_demo

DEFAULT_VIDEO = PROJECT_ROOT / "data" / "raw" / "test1.mp4"

_PRESET = accuracy_yolo_config()


def parse_args():
    p = argparse.ArgumentParser(
        description="Demo nhan dien hanh vi (preset CHINH XAC: detect + skeleton + ST-GCN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Mac dinh da toi uu cho bai toan skeleton-based. "
            "Chi can --video; khong can chinh nhieu tham so."
        ),
    )
    p.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Thu muc ket qua (mac dinh: results/demo_check/<ten_video>_<timestamp>/)",
    )
    p.add_argument("--output", type=str, default=None, help="Ghi de duong dan video output")
    p.add_argument("--max-frames", type=int, default=0, help="0 = het video")
    p.add_argument("--no-skeleton", action="store_true", help="Khong ve khung xuong")
    return p.parse_args()


def resolve_step3_output(args) -> tuple[Path, Path]:
    video = Path(args.video)
    if args.output:
        out = Path(args.output)
        run_dir = out.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        out = step3_output_path(run_dir)
    else:
        run_dir = make_demo_check_run_dir(video)
        out = step3_output_path(run_dir)
    return run_dir, out


def main():
    args = parse_args()
    run_dir, out_video = resolve_step3_output(args)
    cfg = DemoConfig(
        video_path=Path(args.video),
        output_video=out_video,
        max_frames=args.max_frames,
        draw_skeleton=not args.no_skeleton,
    )

    print("=== Demo CHINH XAC (YOLOv8m + Skeleton + ST-GCN) ===")
    print(f"YOLO   : {cfg.yolo_model} | conf={_PRESET.conf} | imgsz={_PRESET.imgsz}")
    print(f"Vote   : {cfg.min_agree}/{cfg.smooth_votes} | ST-GCN min_conf={cfg.min_conf}")
    print(f"Video  : {cfg.video_path}")
    print(f"Run dir: {run_dir}")
    print(f"Output : {cfg.output_video}")

    def on_progress(frac: float, msg: str) -> None:
        if frac >= 0.05:
            print(f"  [{frac * 100:.0f}%] {msg}", flush=True)

    result = run_classroom_demo(cfg, progress=on_progress)

    print(f"\nHoan thanh trong {result.elapsed_sec:.0f}s")
    print(f"Video : {result.output_video}")
    print(f"Tong SV (thong ke): {result.total_students}")
    print(f"Tracks: {result.tracks_detected}")


if __name__ == "__main__":
    main()
