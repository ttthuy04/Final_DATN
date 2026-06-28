"""
Web server: giao dien + chay pipeline demo_check (YOLO + Skeleton + ST-GCN).

Chay:
  python web/server.py
  Mo trinh duyet: http://127.0.0.1:5050
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_file, send_from_directory

from src.common.demo_check_paths import (
    DEMO_CHECK_ROOT,
    make_demo_check_run_dir,
    step3_output_path,
)
from src.common.demo_result_io import (
    finalize_web_run_dir,
    list_demo_runs,
    load_demo_result,
    result_for_api,
)
from src.common.video_web import ensure_web_video
from src.demo.classroom_demo import DemoConfig, run_classroom_demo

WEB_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _safe_unlink(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _run_analysis(
    job_id: str,
    run_dir: Path,
    video_path: Path,
    *,
    draw_skeleton: bool,
    max_frames: int,
    source_name: str,
) -> None:
    try:
        out_video = step3_output_path(run_dir)
        cfg = DemoConfig(
            video_path=video_path,
            output_video=out_video,
            max_frames=max_frames,
            draw_skeleton=draw_skeleton,
        )

        def on_progress(frac: float, msg: str) -> None:
            _set_job(
                job_id,
                status="running",
                progress=round(min(max(frac, 0.0), 1.0) * 100, 1),
                message=msg,
            )

        _set_job(job_id, status="running", progress=2, message="Dang load model...")
        result = run_classroom_demo(cfg, progress=on_progress)

        _set_job(job_id, progress=98, message="Dang chuyen video cho trinh duyet...")
        ensure_web_video(run_dir)
        finalize_web_run_dir(run_dir)

        payload = result_for_api(
            run_dir,
            result,
            source_video=source_name,
            web_output_only=True,
        )

        _set_job(
            job_id,
            status="done",
            progress=100,
            message="Hoan thanh!",
            run_name=run_dir.name,
            result=payload,
        )
    except Exception as exc:
        _set_job(job_id, status="error", message=str(exc), error=str(exc))
    finally:
        _safe_unlink(video_path)


@app.get("/api/health")
def health():
    all_runs = list_demo_runs()
    return jsonify(
        {
            "status": "ok",
            "saved_runs": len(all_runs),
            "latest_run": all_runs[0]["run_name"] if all_runs else None,
        }
    )


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.post("/api/analyze")
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "Thieu file video"}), 400

    upload = request.files["video"]
    if not upload.filename:
        return jsonify({"error": "Ten file khong hop le"}), 400

    draw_skeleton = request.form.get("draw_skeleton", "true").lower() in ("1", "true", "yes")
    try:
        max_frames = int(request.form.get("max_frames", "0") or "0")
    except ValueError:
        max_frames = 0

    job_id = uuid.uuid4().hex
    run_dir = make_demo_check_run_dir(upload.filename)

    suffix = Path(upload.filename).suffix or ".mp4"
    fd, temp_name = tempfile.mkstemp(suffix=suffix, prefix=f"sbr_upload_{job_id}_")
    os.close(fd)
    saved_video = Path(temp_name)
    upload.save(saved_video)
    _set_job(
        job_id,
        status="queued",
        progress=0,
        message="Da nhan video, dang xep hang...",
        run_dir=str(run_dir),
        run_name=run_dir.name,
    )

    worker = threading.Thread(
        target=_run_analysis,
        args=(job_id, run_dir, saved_video),
        kwargs={
            "draw_skeleton": draw_skeleton,
            "max_frames": max_frames,
            "source_name": upload.filename,
        },
        daemon=True,
    )
    worker.start()

    return jsonify({"job_id": job_id, "run_name": run_dir.name})


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Khong tim thay job"}), 404
    return jsonify(job)


@app.get("/api/runs")
def runs():
    return jsonify({"runs": list_demo_runs()})


@app.get("/api/runs/<run_name>")
def run_detail(run_name: str):
    run_dir = DEMO_CHECK_ROOT / run_name
    if not run_dir.is_dir():
        return jsonify({"error": "Khong tim thay ket qua"}), 404

    data = load_demo_result(run_dir)
    if data is None:
        return jsonify({"error": "Chua co result.json"}), 404

    from src.common.demo_result_io import _with_media_urls

    return jsonify(_with_media_urls(run_name, data, prepare_web=True))


@app.get("/files/<run_name>/<path:filename>")
def serve_run_file(run_name: str, filename: str):
    run_dir = (DEMO_CHECK_ROOT / run_name).resolve()
    if not str(run_dir).startswith(str(DEMO_CHECK_ROOT.resolve())):
        return jsonify({"error": "Duong dan khong hop le"}), 403
    if not run_dir.is_dir():
        return jsonify({"error": "Khong tim thay thu muc"}), 404

    file_path = (run_dir / filename).resolve()
    if not str(file_path).startswith(str(run_dir)):
        return jsonify({"error": "Duong dan khong hop le"}), 403
    if not file_path.is_file():
        return jsonify({"error": "Khong tim thay file"}), 404

    mimetype = "video/mp4" if file_path.suffix.lower() == ".mp4" else None
    return send_file(file_path, mimetype=mimetype, conditional=True)


if __name__ == "__main__":
    import webbrowser

    DEMO_CHECK_ROOT.mkdir(parents=True, exist_ok=True)
    print("=== Student Behavior Recognition Web ===")
    print("Pipeline: YOLOv8m + Skeleton + ST-GCN (preset chinh xac)")
    print("Mo trinh duyet: http://127.0.0.1:5050")
    print("Ket qua luu tai: results/demo_check/ (classroom_demo_final_web.mp4 + result.json)")
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5050")).start()
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
