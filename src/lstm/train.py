"""
Train LSTM (optimized):
- custom class weights (normal, use_phone)
- Adam + weight_decay, ReduceLROnPlateau, early stopping, grad clip

Chay: python src/lstm/train.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import ModelPaths
from src.common.training_log import (
    append_log_csv,
    compute_classification_metrics,
    init_log_csv,
    print_log_header,
    print_log_row,
)
from src.dataset.skeleton_dataset import CLASS_NAMES, get_dataloaders
from src.lstm.model import SkeletonLSTM

PATHS = ModelPaths("lstm")

MAX_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.001
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

WEIGHT_BY_CLASS = {
    "normal": 1.6,
    "sleep": 1.0,
    "standing": 1.0,
    "use_phone": 1.3,
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_preds = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return total_loss / len(all_labels), np.array(all_labels), np.array(all_preds)


def main():
    PATHS.ensure_dirs()
    device = get_device()
    print("=== LSTM (optimized) ===", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}", flush=True)

    train_loader, val_loader, _, split_info = get_dataloaders(batch_size=BATCH_SIZE)
    print(
        f"Split -> train: {split_info['train']} | "
        f"val: {split_info['val']} | test: {split_info['test']}",
        flush=True,
    )

    missing = [c for c in CLASS_NAMES if c not in WEIGHT_BY_CLASS]
    if missing:
        raise ValueError(f"Thieu class weight cho: {missing}")
    weight_list = [WEIGHT_BY_CLASS[c] for c in CLASS_NAMES]
    class_weights = torch.tensor(weight_list, dtype=torch.float32).to(device)
    print(f"Class weights: {dict(zip(CLASS_NAMES, weight_list))}", flush=True)

    model = SkeletonLSTM(
        num_classes=len(CLASS_NAMES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    print(f"Params: {model.count_parameters():,}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    init_log_csv(PATHS.training_log)
    print_log_header()
    print(
        f"Config: max_epochs={MAX_EPOCHS} | lr={LR} | weight_decay={WEIGHT_DECAY} | "
        f"grad_clip={GRAD_CLIP} | early_stop={EARLY_STOP_PATIENCE} | "
        f"scheduler patience={SCHEDULER_PATIENCE}",
        flush=True,
    )

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    prev_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_val, y_pred = run_epoch(model, val_loader, criterion, device)
        metrics = compute_classification_metrics(y_val, y_pred)

        print_log_row(epoch, train_loss, val_loss, metrics)
        append_log_csv(PATHS.training_log, epoch, train_loss, val_loss, metrics)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < prev_lr:
            print(f"  -> Giam LR: {prev_lr:.2e} -> {current_lr:.2e}", flush=True)
            prev_lr = current_lr

        if metrics["Accuracy"] > best_val_acc + EARLY_STOP_MIN_DELTA:
            best_val_acc = metrics["Accuracy"]
            best_val_f1 = metrics["F1"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_type": "lstm",
                    "variant": "custom_weights_optimized",
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "class_weights": weight_list,
                    "best_val_acc": best_val_acc,
                    "best_val_f1": best_val_f1,
                    "epoch": epoch,
                    "num_params": model.count_parameters(),
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                },
                PATHS.model_path,
            )
            print(
                f"  -> Luu model tot nhat (val acc={best_val_acc:.4f}, f1={best_val_f1:.4f})",
                flush=True,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping tai epoch {epoch} "
                f"(khong cai thien val acc trong {EARLY_STOP_PATIENCE} epoch)",
                flush=True,
            )
            break

    print(
        f"\nBest epoch: {best_epoch} | val acc={best_val_acc:.4f} | val f1={best_val_f1:.4f}",
        flush=True,
    )
    print(f"Model: {PATHS.model_path}", flush=True)
    print(f"Training log: {PATHS.training_log}", flush=True)
    print("Tiep theo: python src/lstm/evaluate.py", flush=True)
    print("Ve chart: python src/common/plot_history.py --model lstm", flush=True)


if __name__ == "__main__":
    main()
