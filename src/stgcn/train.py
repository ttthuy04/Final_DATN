"""
Train ST-GCN theo bai bao Yan et al. AAAI 2018 + cai tien da chung minh:
- Spatial configuration partitioning (3 subgraph)
- Learnable edge importance M
- Velocity + root normalization
- Custom class weights

Chay: python src/stgcn/train.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import ModelPaths
from src.dataset.skeleton_dataset import CLASS_NAMES, get_dataloaders
from src.stgcn.model import SkeletonSTGCN
from src.stgcn.train_loop import BATCH_SIZE, DROPOUT, get_device, train_stgcn

PATHS = ModelPaths("stgcn")

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

    train_loader, val_loader, _, info = get_dataloaders(batch_size=BATCH_SIZE)
    print(
        f"Split -> train: {info['train']} | val: {info['val']} | test: {info['test']}",
        flush=True,
    )

    missing = [c for c in CLASS_NAMES if c not in WEIGHT_BY_CLASS]
    if missing:
        raise ValueError(f"Thieu class weight cho: {missing}")
    weight_list = [WEIGHT_BY_CLASS[c] for c in CLASS_NAMES]
    class_weights = torch.tensor(weight_list, dtype=torch.float32)
    print(f"Class weights: {dict(zip(CLASS_NAMES, weight_list))}", flush=True)

    model = SkeletonSTGCN(
        num_classes=len(CLASS_NAMES),
        use_velocity=True,
        normalize_root=True,
        use_spatial_partition=True,
        dropout=DROPOUT,
    ).to(device)
    print(
        f"Input channels: {model.in_channels} | Params: {model.count_parameters():,}",
        flush=True,
    )

    best = train_stgcn(
        model,
        train_loader,
        val_loader,
        class_weights,
        PATHS.model_path,
        PATHS.training_log,
        CLASS_NAMES,
        extra_ckpt={
            "variant": "yan2018_spatial_partition",
            "paper": "Yan et al., Spatial Temporal Graph Convolutional Networks, AAAI 2018",
            "use_velocity": True,
            "normalize_root": True,
            "use_spatial_partition": True,
            "learnable_edge_importance": True,
            "class_weights": weight_list,
        },
        device=device,
    )

    print(f"\nHoan thanh. Best val acc: {best:.4f}", flush=True)
    print(f"Model: {PATHS.model_path}", flush=True)
    print(f"Training log: {PATHS.training_log}", flush=True)
    print("Tiep theo: python src/stgcn/evaluate.py", flush=True)
    print("Ve chart: python src/common/plot_history.py --model stgcn", flush=True)


if __name__ == "__main__":
    main()
