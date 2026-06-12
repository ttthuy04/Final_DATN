"""
Pipeline demo lop hoc: YOLO track -> MediaPipe skeleton -> ST-GCN -> video + thong ke.

Dung cho: script/step3_classroom_demo.py (CLI)
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common.paths import ModelPaths
from src.common.demo_presets import (
    ACCURACY_BBOX_PADDING,
    ACCURACY_FAST_VOTE_EVERY,
    ACCURACY_MIN_AGREE,
    ACCURACY_MIN_BEHAVIOR_CONF,
    ACCURACY_MP_DETECTION,
    ACCURACY_MP_PRESENCE,
    ACCURACY_MP_TRACKING,
    ACCURACY_PREDICT_EVERY,
    ACCURACY_SMOOTH_VOTES,
    ACCURACY_YOLO_MODEL,
    accuracy_yolo_config,
)
from src.common.yolo_person_filter import (
    extract_person_boxes,
    filter_person_boxes,
    passes_pose_gate,
    skeleton_is_person_like,
    yolo_predict_kwargs,
)
from src.preprocess.extract_skeleton import N_FRAMES_DEFAULT, get_model_path, landmarks_to_vector
from src.stgcn.model import SkeletonSTGCN
from src.visualization.skeleton_draw import draw_pipeline_legend, draw_skeleton_from_vec

WINDOW = N_FRAMES_DEFAULT

LABEL_VI = {
    "normal": "Binh thuong",
    "sleep": "Ngu gat",
    "standing": "Dung day",
    "use_phone": "Dung dien thoai",
}

LABEL_COLOR = {
    "normal": (0, 200, 0),
    "sleep": (0, 0, 255),
    "standing": (255, 160, 0),
    "use_phone": (0, 220, 255),
}

ProgressCallback = Callable[[float, str], None]


@dataclass
class DemoConfig:
    """Mac dinh = preset chinh xac (xem src/common/demo_presets.py)."""
    video_path: Path
    output_video: Path
    yolo_model: str = ACCURACY_YOLO_MODEL
    conf: float = 0.10
    imgsz: int = 1280
    padding: float = ACCURACY_BBOX_PADDING
    min_box_h: int = 25
    use_pose_gate: bool = True
    require_quality_skeleton: bool = True
    predict_every: int = ACCURACY_PREDICT_EVERY
    fast_vote_every: int = ACCURACY_FAST_VOTE_EVERY
    smooth_votes: int = ACCURACY_SMOOTH_VOTES
    min_agree: int = ACCURACY_MIN_AGREE
    min_conf: float = ACCURACY_MIN_BEHAVIOR_CONF
    max_frames: int = 0
    auto_short_video: bool = True
    draw_skeleton: bool = True

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)
        self.output_video = Path(self.output_video)

    def yolo_filter_config(self):
        cfg = accuracy_yolo_config()
        cfg.conf = self.conf
        cfg.imgsz = self.imgsz
        cfg.min_box_h = self.min_box_h
        return cfg


@dataclass
class DemoResult:
    output_video: Path
    class_names: list[str]
    stats_by_class: dict[str, int]
    stats_table: list[list]
    total_students: int
    tracks_detected: int
    frames_processed: int
    elapsed_sec: float
    last_frame_counts: dict[str, int] = field(default_factory=dict)
    track_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_video": str(self.output_video),
            "class_names": self.class_names,
            "stats_by_class": self.stats_by_class,
            "stats_table": self.stats_table,
            "total_students": self.total_students,
            "tracks_detected": self.tracks_detected,
            "frames_processed": self.frames_processed,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "last_frame_counts": self.last_frame_counts,
            "track_details": self.track_details,
        }


@dataclass
class TrackState:
    buffer: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    label: str = ""
    conf: float = 0.0
    recent_preds: deque = field(default_factory=lambda: deque(maxlen=7))


def create_image_landmarker(model_path: Path) -> PoseLandmarker:
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=ACCURACY_MP_DETECTION,
        min_pose_presence_confidence=ACCURACY_MP_PRESENCE,
        min_tracking_confidence=ACCURACY_MP_TRACKING,
    )
    return PoseLandmarker.create_from_options(opts)


def load_stgcn(device: torch.device):
    paths = ModelPaths("stgcn")
    ckpt_path = paths.model_path
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Chua co ST-GCN tai {ckpt_path}. Hay chay: python src/stgcn/train.py"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    names = ckpt["class_names"]
    model = SkeletonSTGCN(
        num_classes=len(names),
        dropout=ckpt.get("dropout", 0.25),
        use_velocity=ckpt.get("use_velocity", True),
        normalize_root=ckpt.get("normalize_root", True),
        use_spatial_partition=ckpt.get("use_spatial_partition", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, names


@torch.no_grad()
def predict_behavior(
    model, class_names: list[str], seq: np.ndarray, device: torch.device
) -> tuple[str, float, np.ndarray]:
    x = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32, device=device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx]), probs


def expand_bbox(x1, y1, x2, y2, w, h, pad: float):
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0, int(x1 - pad * bw)),
        max(0, int(y1 - pad * bh)),
        min(w, int(x2 + pad * bw)),
        min(h, int(y2 + pad * bh)),
    )


def skeleton_from_crop(landmarker, crop_bgr) -> np.ndarray | None:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_img)
    if not res.pose_landmarks:
        return None
    return np.array(landmarks_to_vector(res.pose_landmarks[0]), dtype=np.float32)


def stable_label(track: TrackState, min_agree: int) -> str | None:
    if len(track.recent_preds) < min_agree:
        return None
    counts: dict[str, int] = defaultdict(int)
    for lbl in track.recent_preds:
        counts[lbl] += 1
    best, cnt = max(counts.items(), key=lambda x: x[1])
    if cnt >= min_agree:
        return best
    return None


def stats_panel_height(class_names: list[str]) -> int:
    return 36 + 26 * len(class_names) + 10


def draw_stats_panel(frame, counts: dict[str, int], class_names: list[str]) -> None:
    panel_w = 310
    x0, y0 = 10, 10
    panel_h = stats_panel_height(class_names)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    cv2.putText(
        frame,
        "THONG KE LOP (%)",
        (x0 + 10, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    total = sum(counts.values())
    for i, cls in enumerate(class_names):
        n = counts.get(cls, 0)
        pct = 100.0 * n / total if total > 0 else 0.0
        vi = LABEL_VI.get(cls, cls)
        col = LABEL_COLOR.get(cls, (255, 255, 255))
        cv2.putText(
            frame,
            f"{vi}: {pct:.1f}%",
            (x0 + 10, y0 + 50 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            2,
        )


def _apply_short_video_tuning(cfg: DemoConfig, total_frames: int, fps: float) -> DemoConfig:
    if not cfg.auto_short_video:
        return cfg
    duration = total_frames / fps if fps > 0 else 0
    if duration <= 12:
        tuned = DemoConfig(**{**asdict(cfg), "auto_short_video": False})
        tuned.predict_every = min(cfg.predict_every, 12)
        tuned.fast_vote_every = min(cfg.fast_vote_every, 5)
        tuned.smooth_votes = min(cfg.smooth_votes, 3)
        tuned.min_agree = min(cfg.min_agree, 2)
        return tuned
    return cfg


def should_predict_behavior(tr: TrackState, frame_idx: int, cfg: DemoConfig) -> bool:
    """Du 60 frame skeleton; du doan ngay lan dau, sau do nhanh cho den khi du vote."""
    if len(tr.buffer) != WINDOW:
        return False
    if not tr.recent_preds:
        return True
    if len(tr.recent_preds) < cfg.min_agree:
        return frame_idx % cfg.fast_vote_every == 0
    return frame_idx % cfg.predict_every == 0


def update_track_label(tr: TrackState, lbl: str, conf: float, cfg: DemoConfig) -> None:
    """Hien nhan tam ngay lan du doan dau; vote de on dinh sau."""
    tr.recent_preds.append(lbl)
    stable = stable_label(tr, cfg.min_agree)
    if stable is not None:
        tr.label = stable
        tr.conf = conf
    elif not tr.label:
        tr.label = lbl
        tr.conf = conf


def aggregate_stats_from_tracks(
    tracks: dict[int, TrackState], class_names: list[str]
) -> dict[str, int]:
    """Moi track_id lay nhan cuoi cung da on dinh."""
    counts = {c: 0 for c in class_names}
    for tr in tracks.values():
        if tr.label:
            counts[tr.label] = counts.get(tr.label, 0) + 1
    return counts


def aggregate_stats_from_log(
    log_rows: list[dict], class_names: list[str]
) -> dict[str, int]:
    """Moi track_id vote nhan xuat hien nhieu nhat trong log."""
    by_track: dict[int, list[str]] = defaultdict(list)
    for row in log_rows:
        lbl = str(row.get("label", "")).strip()
        if lbl and lbl in class_names:
            by_track[int(row["track_id"])].append(lbl)

    counts = {c: 0 for c in class_names}
    for labels in by_track.values():
        if not labels:
            continue
        best = Counter(labels).most_common(1)[0][0]
        counts[best] += 1
    return counts


def build_track_details(
    tracks: dict[int, TrackState],
    log_rows: list[dict],
) -> list[dict]:
    """Tom tat tung track: nhan on dinh + so lan du doan trong log."""
    pred_frames: dict[int, int] = defaultdict(int)
    for row in log_rows:
        pred_frames[int(row["track_id"])] += 1

    details: list[dict] = []
    for tid, tr in sorted(tracks.items()):
        if not tr.label:
            continue
        display_id = tid if tid < 10000 else tid - 10000
        details.append(
            {
                "track_id": display_id,
                "label": tr.label,
                "conf": round(tr.conf, 4),
                "pred_frames": pred_frames.get(tid, 0),
            }
        )
    return details


def build_stats_table(
    stats: dict[str, int], class_names: list[str]
) -> tuple[list[list], int]:
    total = sum(stats.values())
    rows: list[list] = []
    for cls in class_names:
        n = stats.get(cls, 0)
        pct = f"{100.0 * n / total:.1f}%" if total > 0 else "0.0%"
        rows.append([LABEL_VI.get(cls, cls), cls, n, pct])
    rows.append(["Tong sinh vien", "—", total, "100.0%" if total > 0 else "0.0%"])
    return rows, total


def _report_progress(cb: ProgressCallback | None, frac: float, msg: str) -> None:
    if cb is not None:
        cb(min(max(frac, 0.0), 1.0), msg)


def run_classroom_demo(
    config: DemoConfig,
    progress: ProgressCallback | None = None,
) -> DemoResult:
    if not config.video_path.exists():
        raise FileNotFoundError(f"Khong thay video: {config.video_path}")

    config.output_video.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _report_progress(progress, 0.02, "Dang load ST-GCN...")

    stgcn, class_names = load_stgcn(device)
    yolo = YOLO(config.yolo_model)
    landmarker = create_image_landmarker(get_model_path(PROJECT_ROOT))

    cap = cv2.VideoCapture(str(config.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if config.max_frames > 0:
        total = min(total, config.max_frames)

    cfg = _apply_short_video_tuning(config, total, fps)

    writer = cv2.VideoWriter(
        str(cfg.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        cap.release()
        landmarker.close()
        raise RuntimeError(f"Khong ghi duoc video output: {cfg.output_video}")

    tracks: dict[int, TrackState] = {}
    log_rows: list[dict] = []
    last_frame_counts: dict[str, int] = {}
    frame_idx = 0
    t0 = time.time()

    _report_progress(progress, 0.05, "Dang xu ly video...")

    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break

        yolo_cfg = cfg.yolo_filter_config()
        yolo_kw = yolo_predict_kwargs(yolo_cfg)

        results = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            **yolo_kw,
        )[0]
        raw_boxes = extract_person_boxes(results, h, w)
        geo_boxes = filter_person_boxes(
            raw_boxes, h, w, yolo_cfg, use_pose_gate=False
        )
        frame_vis = frame.copy()
        frame_counts: dict[str, int] = defaultdict(int)

        for pi, pbox in enumerate(geo_boxes):
            tid = pbox.track_id
            if tid < 0:
                tid = 10000 + pi
            x1, y1, x2, y2 = pbox.x1, pbox.y1, pbox.x2, pbox.y2
            ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, w, h, cfg.padding)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue

            vec = skeleton_from_crop(landmarker, crop)
            if cfg.use_pose_gate and not passes_pose_gate(pbox, vec, yolo_cfg):
                continue

            crop_h, crop_w = crop.shape[:2]
            skel_ok = vec is not None and skeleton_is_person_like(vec, yolo_cfg)
            if cfg.draw_skeleton and vec is not None:
                draw_skeleton_from_vec(frame_vis, vec, ex1, ey1, crop_w, crop_h)

            if tid not in tracks:
                tracks[tid] = TrackState()
            tr = tracks[tid]
            if len(tr.recent_preds) == 0:
                tr.recent_preds = deque(maxlen=cfg.smooth_votes)

            if skel_ok or (vec is not None and not cfg.require_quality_skeleton):
                if vec is not None:
                    tr.buffer.append(vec)

            if len(tr.buffer) == WINDOW and should_predict_behavior(tr, frame_idx, cfg):
                seq = np.stack(list(tr.buffer), axis=0)
                lbl, conf, _ = predict_behavior(stgcn, class_names, seq, device)
                if conf >= cfg.min_conf:
                    update_track_label(tr, lbl, conf, cfg)
                    log_rows.append(
                        {
                            "frame": frame_idx,
                            "track_id": tid,
                            "label": tr.label or lbl,
                            "conf": round(conf, 4),
                            "raw_pred": lbl,
                            "yolo_conf": round(pbox.conf, 4),
                            "skeleton_ok": skel_ok,
                        }
                    )

            if tr.label:
                lbl_vi = LABEL_VI.get(tr.label, tr.label)
                color = LABEL_COLOR.get(tr.label, (0, 255, 0))
                frame_counts[tr.label] += 1
            else:
                lbl_vi = "..."
                color = (180, 180, 180)

            cv2.rectangle(frame_vis, (ex1, ey1), (ex2, ey2), color, 2)
            conf_txt = f"{tr.conf * 100:.0f}%" if tr.label else "--"
            yolo_txt = f"Y{pbox.conf * 100:.0f}%"
            txt = f"ID{tid} {lbl_vi} {conf_txt} ({yolo_txt})"
            cv2.putText(
                frame_vis,
                txt,
                (ex1, max(ey1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        last_frame_counts = dict(frame_counts)
        draw_stats_panel(frame_vis, frame_counts, class_names)
        if cfg.draw_skeleton:
            draw_pipeline_legend(frame_vis, y=10 + stats_panel_height(class_names) + 8)
        cv2.putText(
            frame_vis,
            f"Frame {frame_idx + 1}/{total}",
            (20, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(frame_vis)
        frame_idx += 1

        if frame_idx % 5 == 0 or frame_idx == total:
            frac = 0.05 + 0.93 * (frame_idx / max(total, 1))
            _report_progress(progress, frac, f"Dang xu ly frame {frame_idx}/{total}")

    cap.release()
    landmarker.close()
    writer.release()

    stats_log = aggregate_stats_from_log(log_rows, class_names)
    stats_tracks = aggregate_stats_from_tracks(tracks, class_names)
    stats_by_class = stats_log if sum(stats_log.values()) > 0 else stats_tracks
    stats_table, total_students = build_stats_table(stats_by_class, class_names)
    track_details = build_track_details(tracks, log_rows)

    elapsed = time.time() - t0
    _report_progress(progress, 1.0, "Hoan thanh!")

    return DemoResult(
        output_video=cfg.output_video,
        class_names=class_names,
        stats_by_class=stats_by_class,
        stats_table=stats_table,
        total_students=total_students,
        tracks_detected=len(tracks),
        frames_processed=frame_idx,
        elapsed_sec=elapsed,
        last_frame_counts=last_frame_counts,
        track_details=track_details,
    )
