"""
Train LSTM (optimized):
- custom class weights (normal, use_phone)
- Adam + weight_decay, ReduceLROnPlateau, early stopping, grad clip

Chay: python src/lstm/train.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import ModelPaths
from src.common.train_loop import BATCH_SIZE, get_device, train_model
from src.dataset.skeleton_dataset import CLASS_NAMES, get_dataloaders
from src.lstm.model import SkeletonLSTM

PATHS = ModelPaths("lstm")

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3  # 0.3 dropout de tranh overfitting

# Dong bo voi ST-GCN de so sanh cong bang (F1 macro, patience 15)
SAVE_BEST_BY = "f1"
EARLY_STOP_PATIENCE = 15

WEIGHT_BY_CLASS = {
    "normal": 1.6,
    "sleep": 1.0,
    "standing": 1.0,
    "use_phone": 1.3,
}
# giúp mô hình chú ý tới nhãn hay bị sai


def main():
    PATHS.ensure_dirs()
    device = get_device()
    print("=== LSTM (optimized) ===", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}", flush=True)

    train_loader, val_loader, _, split_info = get_dataloaders(batch_size=BATCH_SIZE)
    # bỏ qua test loader , nó dùng trong evaluate , lấy trong data/skeleton
    print(
        f"Split -> train: {split_info['train']} | "
        f"val: {split_info['val']} | test: {split_info['test']}",
        flush=True,
    )

    missing = [c for c in CLASS_NAMES if c not in WEIGHT_BY_CLASS]
    if missing:
        raise ValueError(f"Thieu class weight cho: {missing}")
    weight_list = [WEIGHT_BY_CLASS[c] for c in CLASS_NAMES]
    class_weights = torch.tensor(weight_list, dtype=torch.float32)
    print(f"Class weights: {dict(zip(CLASS_NAMES, weight_list))}", flush=True)

    model = SkeletonLSTM(
        num_classes=len(CLASS_NAMES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    print(f"Params: {model.count_parameters():,}", flush=True)

    best = train_model(
        model,
        train_loader,
        val_loader,
        class_weights,
        PATHS.model_path,
        PATHS.training_log,
        CLASS_NAMES,
        model_type="lstm",
        device=device,
        dropout=DROPOUT,
        save_best_by=SAVE_BEST_BY,
        early_stop_patience=EARLY_STOP_PATIENCE,
        extra_ckpt={
            "variant": "custom_weights_optimized",
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "class_weights": weight_list,
        },
    )

    print(f"\nHoan thanh. Best val acc: {best:.4f}", flush=True)
    print(f"Model: {PATHS.model_path}", flush=True)
    print(f"Training log: {PATHS.training_log}", flush=True)
    print("Tiep theo: python src/lstm/evaluate.py", flush=True)
    print("Ve chart: python src/common/plot_history.py --model lstm", flush=True)


if __name__ == "__main__":
    main()
