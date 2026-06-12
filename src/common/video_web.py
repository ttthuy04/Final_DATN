"""Chuyen video OpenCV (mp4v) sang H.264 de xem tren trinh duyet."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.common.demo_check_paths import step3_output_path

WEB_OUTPUT_NAME = "classroom_demo_final_web.mp4"


def web_output_path(run_dir: Path) -> Path:
    return Path(run_dir) / WEB_OUTPUT_NAME


def transcode_for_web(src: Path, dst: Path | None = None) -> Path | None:
    src = Path(src)
    if not src.exists():
        return None

    dst = Path(dst) if dst is not None else src.parent / WEB_OUTPUT_NAME
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime and dst.stat().st_size > 0:
        return dst

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(dst),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return dst
    except Exception:
        return None
    return None


def ensure_web_video(run_dir: Path) -> Path | None:
    """Tao ban H.264 neu chua co (tu classroom_demo_final.mp4)."""
    run_dir = Path(run_dir)
    web = web_output_path(run_dir)
    if web.exists() and web.stat().st_size > 0:
        return web
    return transcode_for_web(step3_output_path(run_dir), web)
