"""Ve khung xuong len frame / crop — dung chung step2, step3."""
from __future__ import annotations

import cv2
import numpy as np

from src.preprocess.extract_skeleton import N_LANDMARKS
from src.stgcn.model import SKELETON_EDGES

JOINT_COLOR = (0, 0, 255)
BONE_COLOR = (255, 120, 0)
MIN_VIS_DEFAULT = 0.25


def _point_from_vec(vec: np.ndarray, j: int, ox: int, oy: int, cw: int, ch: int, min_vis: float):
    base = j * 4
    x_n = float(vec[base])
    y_n = float(vec[base + 1])
    vis = float(vec[base + 3])
    if vis < min_vis or (x_n <= 0.01 and y_n <= 0.01):
        return None
    return int(ox + x_n * cw), int(oy + y_n * ch)


def _point_from_landmark(lm, ox: int, oy: int, cw: int, ch: int, min_vis: float):
    vis = getattr(lm, "visibility", None)
    if vis is None:
        vis = getattr(lm, "presence", 1.0)
    if float(vis) < min_vis:
        return None
    return int(ox + lm.x * cw), int(oy + lm.y * ch)


def draw_skeleton_from_vec(
    frame: np.ndarray,
    vec: np.ndarray,
    ox: int,
    oy: int,
    cw: int,
    ch: int,
    min_vis: float = MIN_VIS_DEFAULT,
    joint_radius: int = 4,
    bone_thickness: int = 2,
) -> int:
    """Ve skeleton tu vector (100,). Tra ve so khop ve duoc."""
    pts: list[tuple[int, int] | None] = []
    visible = 0
    for j in range(N_LANDMARKS):
        pt = _point_from_vec(vec, j, ox, oy, cw, ch, min_vis)
        pts.append(pt)
        if pt is not None:
            visible += 1

    for a, b in SKELETON_EDGES:
        if a >= len(pts) or b >= len(pts):
            continue
        pa, pb = pts[a], pts[b]
        if pa is not None and pb is not None:
            cv2.line(frame, pa, pb, BONE_COLOR, bone_thickness)

    for pt in pts:
        if pt is not None:
            cv2.circle(frame, pt, joint_radius, JOINT_COLOR, -1)

    return visible


def draw_skeleton_from_landmarks(
    frame: np.ndarray,
    landmarks,
    ox: int,
    oy: int,
    cw: int,
    ch: int,
    min_vis: float = MIN_VIS_DEFAULT,
    joint_radius: int = 4,
    bone_thickness: int = 2,
) -> int:
    pts: list[tuple[int, int] | None] = []
    visible = 0
    n = min(N_LANDMARKS, len(landmarks))
    for j in range(n):
        pt = _point_from_landmark(landmarks[j], ox, oy, cw, ch, min_vis)
        pts.append(pt)
        if pt is not None:
            visible += 1
    for j in range(n, N_LANDMARKS):
        pts.append(None)

    for a, b in SKELETON_EDGES:
        if a >= len(pts) or b >= len(pts):
            continue
        pa, pb = pts[a], pts[b]
        if pa is not None and pb is not None:
            cv2.line(frame, pa, pb, BONE_COLOR, bone_thickness)

    for pt in pts:
        if pt is not None:
            cv2.circle(frame, pt, joint_radius, JOINT_COLOR, -1)

    return visible


def draw_pipeline_legend(frame: np.ndarray, y: int = 50) -> None:
    """Ghi chu nho: pipeline skeleton-based."""
    cv2.rectangle(frame, (10, y), (340, y + 48), (20, 20, 20), -1)
    cv2.circle(frame, (24, y + 16), 5, JOINT_COLOR, -1)
    cv2.line(frame, (36, y + 16), (52, y + 16), BONE_COLOR, 2)
    cv2.putText(
        frame,
        "Khung xuong (25 khop) -> ST-GCN",
        (58, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        frame,
        "Do: khop | Cam: xuong",
        (58, y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1,
    )
