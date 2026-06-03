"""
Train 2s-AGCN theo bai bao Shi et al. CVPR 2019:
- Two-stream: Joint + Bone
- Adaptive graph convolution
- Velocity + root normalization
- Custom class weights

Chay: python src/agcn/train.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agcn.model import SkeletonTwoStreamAGCN
from src.agcn.train_loop import BATCH_SIZE, DROPOUT, get_device, train_agcn
from src.common.paths import ModelPaths
from src.dataset.skeleton_dataset import CLASS_NAMES, get_dataloaders

PATHS = ModelPaths("agcn")

WEIGHT_BY_CLASS = {
    "normal": 1.6,
    "sleep": 1.0,
    "standing": 1.0,
    "use_phone": 1.3,
}


def main():
    PATHS.ensure_dirs()
    device = get_device()
    print("=== 2s-AGCN (Shi et al. CVPR 2019) ===", flush=True)
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

    model = SkeletonTwoStreamAGCN(
        num_classes=len(CLASS_NAMES),
        use_velocity=True,
        normalize_root=True,
        dropout=DROPOUT,
    ).to(device)
    print(
        f"Input channels: {model.in_channels} | Params: {model.count_parameters():,}",
        flush=True,
    )

    best = train_agcn(
        model,
        train_loader,
        val_loader,
        class_weights,
        PATHS.model_path,
        PATHS.training_log,
        CLASS_NAMES,
        extra_ckpt={
            "variant": "2s_agcn_joint_bone",
            "paper": "Shi et al., Two-Stream Adaptive Graph Convolutional Networks, CVPR 2019",
            "use_velocity": True,
            "normalize_root": True,
            "two_stream": True,
            "adaptive_graph": True,
            "class_weights": weight_list,
        },
        device=device,
    )

    print(f"\nHoan thanh. Best val acc: {best:.4f}", flush=True)
    print(f"Model: {PATHS.model_path}", flush=True)
    print(f"Training log: {PATHS.training_log}", flush=True)
    print("Tiep theo: python src/agcn/evaluate.py", flush=True)
    print("Ve chart: python src/common/plot_history.py --model agcn", flush=True)


if __name__ == "__main__":
    main()
