"""
Train ST-GCN theo bai bao Yan et al. AAAI 2018 + cai tien da chung minh:
-dùng 3 graph: self, inward, outward
-học trọng số cạnh M
-có thêm volocity và root normalization
-CÓ CLASS WEIGHT ĐỂ XỬ LÝ LỆCH KHỚP 
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import ModelPaths #đường dẫn lưu model và log
from src.common.train_loop import BATCH_SIZE, get_device, train_model
from src.dataset.skeleton_dataset import CLASS_NAMES, get_dataloaders #
from src.stgcn.model import SkeletonSTGCN

PATHS = ModelPaths("stgcn")

DROPOUT = 0.25  # tắt ngẫu nhiên 1 phần neuron để tránh overfitting

# Luu best theo F1 macro (on dinh hon accuracy khi val dao dong)
SAVE_BEST_BY = "f1"
EARLY_STOP_PATIENCE = 15

WEIGHT_BY_CLASS = {
    "normal": 1.6,
    "sleep": 1.0,
    "standing": 1.0,
    "use_phone": 1.3,
}


def main():
    PATHS.ensure_dirs()
    device = get_device()
    print("=== ST-GCN (Yan et al. AAAI 2018) ===", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}", flush=True)

    train_loader, val_loader, _, info = get_dataloaders(batch_size=BATCH_SIZE) #lấy dữ liệu train/val/test nhủng bỏ testloader
    print(
        f"Split -> train: {info['train']} | val: {info['val']} | test: {info['test']}",
        flush=True,
    )

    missing = [c for c in CLASS_NAMES if c not in WEIGHT_BY_CLASS]
    if missing:
        raise ValueError(f"Thieu class weight cho: {missing}")
    weight_list = [WEIGHT_BY_CLASS[c] for c in CLASS_NAMES] #tạo list trọng số cho từng class
    class_weights = torch.tensor(weight_list, dtype=torch.float32) #đổi thành tensor
    print(f"Class weights: {dict(zip(CLASS_NAMES, weight_list))}", flush=True)

    model = SkeletonSTGCN(
        num_classes=len(CLASS_NAMES), #4 hành vi
        use_velocity=True,
        normalize_root=True, # chuẩn hóa tọa độ trung tâm của vai
        use_spatial_partition=True, # sử dụng spatial configuration partitioning
        dropout=DROPOUT, # dropout để tránh overfitting
    ).to(device)
    print(
        f"Input channels: {model.in_channels} | Params: {model.count_parameters():,}",
        flush=True,
    )
# bắt đầu huấn luyện
    best = train_model(
        model,
        train_loader,
        val_loader,
        class_weights,
        PATHS.model_path,
        PATHS.training_log,
        CLASS_NAMES,
        model_type="stgcn",
        device=device,
        dropout=DROPOUT,
        save_best_by=SAVE_BEST_BY,
        early_stop_patience=EARLY_STOP_PATIENCE,
        extra_ckpt={
            "variant": "yan2018_spatial_partition",
            "paper": "Yan et al., Spatial Temporal Graph Convolutional Networks, AAAI 2018",
            "use_velocity": True,
            "normalize_root": True,
            "use_spatial_partition": True,
            "learnable_edge_importance": True,
            "class_weights": weight_list,
        },
    )

    print(f"\nHoan thanh. Best val acc: {best:.4f}", flush=True)
    print(f"Model: {PATHS.model_path}", flush=True)
    print(f"Training log: {PATHS.training_log}", flush=True)
    print("Tiep theo: python src/stgcn/evaluate.py", flush=True)
    print("Ve chart: python src/common/plot_history.py --model stgcn", flush=True)


if __name__ == "__main__":
    main()
