"""
Cau hinh mac dinh duy nhat — uu tien chinh xac (detect + skeleton + hanh vi).

Dung cho: step1, step2, step3.
Khong can chon nhieu mode — chi can chay script voi --video.
"""
from __future__ import annotations

from src.common.yolo_person_filter import YoloPersonFilterConfig

# YOLO: can bang tot nhat cho CPU (tot hon n/s, nhanh hon x)
ACCURACY_YOLO_MODEL = "yolov8m.pt"


def accuracy_yolo_config() -> YoloPersonFilterConfig:
    """
    Detect nguoi (ke ca guc ban):
    - conf thap + imgsz lon -> recall
    - loc hinh hoc + pose gate -> giam chan ban/ghe
    """
    return YoloPersonFilterConfig(
        conf=0.10,
        imgsz=1280,
        iou=0.45,
        max_det=35,
        min_box_h=25,
        min_box_w=20,
        min_box_area=600,
        max_h_over_w=5.2,
        min_h_over_w=0.15,
        max_w_over_h=4.5,
        bottom_margin_ratio=0.97,
        bottom_strip_max_h=45,
        bottom_strip_max_w=45,
        thin_leg_max_w=18,
        thin_leg_min_h=80,
        high_conf_trust=0.40,
        pose_verify_below_conf=0.30,
        min_pose_landmarks=6,
        min_pose_visibility=0.20,
        min_torso_landmarks=1,
        slumped_max_h_over_w=1.40,
        slumped_min_w=25,
        slumped_min_h=20,
        slumped_max_h=150,
    )


# ST-GCN / vote: on dinh hon, tranh nhãn nhảy sai
ACCURACY_PREDICT_EVERY = 20
ACCURACY_FAST_VOTE_EVERY = 10
ACCURACY_SMOOTH_VOTES = 5
ACCURACY_MIN_AGREE = 3
ACCURACY_MIN_BEHAVIOR_CONF = 0.55
ACCURACY_BBOX_PADDING = 0.18

# Gop track theo vi tri (1 vi tri = 1 sinh vien):
# - ByteTrack co the doi track_id cho cung 1 nguoi (bi che, ra vao khung) -> dem trung.
# - Gan moi detection vao mot "seat" theo track_id (uu tien) + IoU vi tri (du phong)
#   nen 1 cho ngoi chi tinh 1 sinh vien du track_id thay doi.
ACCURACY_SEAT_MATCH_IOU = 0.35  # IoU toi thieu de coi detection thuoc ve seat da co
ACCURACY_MIN_SEAT_FRAMES = 3  # seat xuat hien it hon nguong nay (va chua co nhan) -> bo (nhieu)

# MediaPipe tren crop (chat luong skeleton cho ST-GCN)
ACCURACY_MP_DETECTION = 0.55
ACCURACY_MP_PRESENCE = 0.55
ACCURACY_MP_TRACKING = 0.55
