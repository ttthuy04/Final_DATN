"""Duong dan output pipeline demo_check — moi lan chay tao thu muc rieng, tranh ghi de."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.common.paths import PROJECT_ROOT

DEMO_CHECK_ROOT = PROJECT_ROOT / "results" / "demo_check"


def make_demo_check_run_dir(
    video_path: Path | str,
    *,
    base: Path | None = None,
    run_name: str | None = None,
) -> Path:
    """Tao thu muc con theo ten video + timestamp."""
    video_path = Path(video_path)
    root = base if base is not None else DEMO_CHECK_ROOT
    stem = video_path.stem or "video"
    if run_name:
        name = run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stem}_{stamp}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def step1_output_path(run_dir: Path) -> Path:
    return run_dir / "step1_yolo_output.mp4"


def step2_paths(run_dir: Path) -> tuple[Path, Path]:
    return run_dir / "step2_crops", run_dir / "step2_skeleton_preview.mp4"


def step3_output_path(run_dir: Path) -> Path:
    return run_dir / "classroom_demo_final.mp4"


def input_video_path(run_dir: Path) -> Path:
    return run_dir / "input_video.mp4"


def result_json_path(run_dir: Path) -> Path:
    return run_dir / "result.json"
