"""Luu / doc ket qua demo_check (result.json)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.demo_check_paths import DEMO_CHECK_ROOT, result_json_path
from src.common.video_web import WEB_OUTPUT_NAME, ensure_web_video, web_output_path
from src.demo.classroom_demo import DemoResult


def save_demo_result(
    run_dir: Path,
    result: DemoResult,
    *,
    source_video: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    payload: dict[str, Any] = {
        **result.to_dict(),
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "source_video": source_video or "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = result_json_path(run_dir)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_demo_result(run_dir: Path) -> dict[str, Any] | None:
    path = result_json_path(Path(run_dir))
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_demo_runs(*, limit: int = 50) -> list[dict[str, Any]]:
    """Liet ke cac lan chay trong results/demo_check co result.json."""
    if not DEMO_CHECK_ROOT.exists():
        return []

    runs: list[dict[str, Any]] = []
    for child in DEMO_CHECK_ROOT.iterdir():
        if not child.is_dir():
            continue
        data = load_demo_result(child)
        if data is None:
            continue
        runs.append(_with_media_urls(child.name, data))

    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs[:limit]


def _output_video_filename(run_dir: Path) -> str | None:
    run_dir = Path(run_dir)
    web = web_output_path(run_dir)
    if web.exists() and web.stat().st_size > 0:
        return WEB_OUTPUT_NAME
    raw = run_dir / "classroom_demo_final.mp4"
    if raw.exists():
        return raw.name
    return None


def _with_media_urls(run_name: str, data: dict[str, Any], *, prepare_web: bool = False) -> dict[str, Any]:
    out = dict(data)
    out["run_name"] = run_name
    run_dir = DEMO_CHECK_ROOT / run_name

    if prepare_web:
        ensure_web_video(run_dir)

    filename = _output_video_filename(run_dir)
    if filename:
        out["output_video_url"] = f"/files/{run_name}/{filename}"

    input_name = "input_video.mp4"
    if (run_dir / input_name).exists():
        out["input_video_url"] = f"/files/{run_name}/{input_name}"
    return out


def result_for_api(run_dir: Path, result: DemoResult, *, source_video: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    save_demo_result(run_dir, result, source_video=source_video)
    data = load_demo_result(run_dir)
    assert data is not None
    return _with_media_urls(run_dir.name, data)
