"""
Wrapper cho web Gradio — xu ly upload, dat ten output, goi pipeline demo.
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.demo.classroom_demo import DemoConfig, DemoResult, run_classroom_demo

WEB_DIR = PROJECT_ROOT / "web"
UPLOAD_DIR = WEB_DIR / "uploads"
OUTPUT_DIR = WEB_DIR / "outputs"
SAMPLE_VIDEO = PROJECT_ROOT / "data" / "raw" / "test1.mp4"

ALLOWED_SUFFIX = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_MB = 200


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _validate_video_path(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Khong tim thay video: {video_path}")
    if video_path.suffix.lower() not in ALLOWED_SUFFIX:
        raise ValueError(f"Dinh dang khong ho tro: {video_path.suffix}. Dung: {ALLOWED_SUFFIX}")
    size_mb = video_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(f"Video qua lon ({size_mb:.1f} MB). Gioi han {MAX_UPLOAD_MB} MB.")


def _copy_upload(src: Path) -> Path:
    ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    dest = UPLOAD_DIR / f"{stamp}_{uid}{src.suffix.lower()}"
    shutil.copy2(src, dest)
    return dest


def _output_paths(stem: str) -> tuple[Path, Path, Path]:
    out_video = OUTPUT_DIR / f"{stem}_result.mp4"
    log_csv = OUTPUT_DIR / f"{stem}_log.csv"
    stats_json = OUTPUT_DIR / f"{stem}_stats.json"
    return out_video, log_csv, stats_json


def run_demo_from_upload(
    video_input: str | Path | None,
    progress_callback=None,
) -> tuple[str | None, list[list], str]:
    """
    Args:
        video_input: duong dan tu Gradio Video upload hoac None.

    Returns:
        (output_video_path, stats_table_rows, status_message)
    """
    if video_input is None or str(video_input).strip() == "":
        return None, [], "Hay upload video hoac chon video mau."

    src = Path(video_input)
    try:
        _validate_video_path(src)
    except (FileNotFoundError, ValueError) as exc:
        return None, [], str(exc)

    stored = _copy_upload(src)
    stem = stored.stem
    out_video, log_csv, stats_json = _output_paths(stem)

    cfg = DemoConfig(
        video_path=stored,
        output_video=out_video,
        log_csv=log_csv,
        stats_json=stats_json,
        auto_short_video=True,
    )

    try:
        result = run_classroom_demo(cfg, progress=progress_callback)
    except FileNotFoundError as exc:
        return None, [], str(exc)
    except Exception as exc:
        return None, [], f"Loi khi xu ly: {exc}"

    status = (
        f"Hoan thanh trong {result.elapsed_sec:.0f}s | "
        f"{result.frames_processed} frame | "
        f"{result.tracks_detected} track | "
        f"{result.total_students} SV co nhan on dinh"
    )
    return str(result.output_video), result.stats_table, status


def get_sample_video_path() -> str | None:
    if SAMPLE_VIDEO.exists():
        return str(SAMPLE_VIDEO)
    alt = PROJECT_ROOT / "data" / "raw" / "classroom_test.mp4"
    if alt.exists():
        return str(alt)
    return None
