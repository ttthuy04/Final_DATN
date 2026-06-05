"""
Loc bbox YOLO class 'person' cho video lop hoc (goc tren).

Muc tieu:
- Giam false positive: chan ban, ghe, vat thang mong
- Van giu nguoi guc ban / ngu (bbox thap, rong)
- Buoc 2/3: xac minh them bang MediaPipe neu conf thap
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PERSON_CLASS = 0

TORSO_LANDMARKS = (11, 12, 23, 24)


@dataclass
class PersonBox:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_track_xyxy(self) -> tuple[int, int, int, int, int]:
        return self.track_id, self.x1, self.y1, self.x2, self.y2


@dataclass
class YoloPersonFilterConfig:
    conf: float = 0.10
    imgsz: int = 1280
    iou: float = 0.45
    max_det: int = 35

    min_box_h: int = 25
    min_box_w: int = 20
    min_box_area: int = 600

    max_h_over_w: float = 5.2
    min_h_over_w: float = 0.15
    max_w_over_h: float = 4.5

    bottom_margin_ratio: float = 0.97
    bottom_strip_max_h: int = 45
    bottom_strip_max_w: int = 45

    thin_leg_max_w: int = 18
    thin_leg_min_h: int = 80

    high_conf_trust: float = 0.40
    pose_verify_below_conf: float = 0.30
    min_pose_landmarks: int = 6
    min_pose_visibility: float = 0.20
    min_torso_landmarks: int = 1

    slumped_max_h_over_w: float = 1.40
    slumped_min_w: int = 25
    slumped_min_h: int = 20
    slumped_max_h: int = 150


def yolo_predict_kwargs(cfg: YoloPersonFilterConfig) -> dict:
    return dict(
        conf=cfg.conf,
        classes=[PERSON_CLASS],
        imgsz=cfg.imgsz,
        iou=cfg.iou,
        max_det=cfg.max_det,
        device="cpu",
        verbose=False,
    )


def extract_person_boxes(results, frame_h: int, frame_w: int) -> list[PersonBox]:
    boxes: list[PersonBox] = []
    if results.boxes is None:
        return boxes
    for box in results.boxes:
        if int(box.cls[0]) != PERSON_CLASS:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tid = int(box.id[0]) if box.id is not None else -1
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(PersonBox(tid, x1, y1, x2, y2, conf))
    return boxes


def _iou(a: PersonBox, b: PersonBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def suppress_overlapping(boxes: list[PersonBox], iou_thresh: float = 0.72) -> list[PersonBox]:
    kept: list[PersonBox] = []
    for box in sorted(boxes, key=lambda b: b.conf, reverse=True):
        if any(_iou(box, k) > iou_thresh for k in kept):
            continue
        kept.append(box)
    return kept


def is_slumped_shape(box: PersonBox, cfg: YoloPersonFilterConfig) -> bool:
    if box.w < cfg.slumped_min_w or box.h < cfg.slumped_min_h:
        return False
    if box.h > cfg.slumped_max_h:
        return False
    h_over_w = box.h / max(box.w, 1)
    return h_over_w <= cfg.slumped_max_h_over_w


def passes_geometry(box: PersonBox, frame_h: int, frame_w: int, cfg: YoloPersonFilterConfig) -> bool:
    if box.h < cfg.min_box_h or box.w < cfg.min_box_w:
        return False
    if box.area < cfg.min_box_area:
        return False

    h_over_w = box.h / max(box.w, 1)
    w_over_h = box.w / max(box.h, 1)

    if h_over_w > cfg.max_h_over_w or h_over_w < cfg.min_h_over_w:
        return False
    if w_over_h > cfg.max_w_over_h:
        return False

    if box.w <= cfg.thin_leg_max_w and box.h >= cfg.thin_leg_min_h:
        return False

    if box.y2 >= int(frame_h * cfg.bottom_margin_ratio):
        if box.h <= cfg.bottom_strip_max_h and box.w <= cfg.bottom_strip_max_w:
            if not is_slumped_shape(box, cfg):
                return False

    return True


def count_valid_landmarks(vec: np.ndarray, min_vis: float) -> int:
    n = 0
    for i in range(25):
        base = i * 4
        vis = float(vec[base + 3])
        x, y = float(vec[base]), float(vec[base + 1])
        if vis >= min_vis and (x > 0.01 or y > 0.01):
            n += 1
    return n


def count_torso_landmarks(vec: np.ndarray, min_vis: float) -> int:
    n = 0
    for i in TORSO_LANDMARKS:
        base = i * 4
        vis = float(vec[base + 3])
        x, y = float(vec[base]), float(vec[base + 1])
        if vis >= min_vis and (x > 0.01 or y > 0.01):
            n += 1
    return n


def skeleton_is_person_like(vec: np.ndarray | None, cfg: YoloPersonFilterConfig) -> bool:
    if vec is None:
        return False
    if count_valid_landmarks(vec, cfg.min_pose_visibility) < cfg.min_pose_landmarks:
        return False
    if count_torso_landmarks(vec, cfg.min_pose_visibility) < cfg.min_torso_landmarks:
        return False
    return True


def passes_pose_gate(
    box: PersonBox,
    skeleton_vec: np.ndarray | None,
    cfg: YoloPersonFilterConfig,
) -> bool:
    if box.conf >= cfg.high_conf_trust:
        return True
    if skeleton_is_person_like(skeleton_vec, cfg):
        return True
    if is_slumped_shape(box, cfg) and box.conf >= cfg.conf:
        return True
    if box.conf < cfg.pose_verify_below_conf:
        return False
    return True


def filter_person_boxes(
    boxes: list[PersonBox],
    frame_h: int,
    frame_w: int,
    cfg: YoloPersonFilterConfig,
    skeleton_by_index: list[np.ndarray | None] | None = None,
    use_pose_gate: bool = False,
) -> list[PersonBox]:
    geo_ok = [b for b in boxes if passes_geometry(b, frame_h, frame_w, cfg)]
    deduped = suppress_overlapping(geo_ok)

    if not use_pose_gate:
        return deduped

    kept: list[PersonBox] = []
    for i, box in enumerate(deduped):
        sk = None
        if skeleton_by_index is not None and i < len(skeleton_by_index):
            sk = skeleton_by_index[i]
        if passes_pose_gate(box, sk, cfg):
            kept.append(box)
    return kept


def filter_yolo_results(
    results,
    frame_h: int,
    frame_w: int,
    cfg: YoloPersonFilterConfig | None = None,
    use_pose_gate: bool = False,
) -> list[PersonBox]:
    cfg = cfg or YoloPersonFilterConfig()
    raw = extract_person_boxes(results, frame_h, frame_w)
    return filter_person_boxes(raw, frame_h, frame_w, cfg, use_pose_gate=use_pose_gate)
