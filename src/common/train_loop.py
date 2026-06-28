"""
Vong lap train dung chung cho ca LSTM va ST-GCN.

Tach rieng de tranh trung lap code giua src/lstm/train.py va src/stgcn/train.py.
Hanh vi (tieu chi luu best, early stopping) duoc truyen vao qua tham so nen moi
model van giu dung cau hinh rieng cua no.
"""

import numpy as np
import torch
import torch.nn as nn

from src.common.training_log import (
    append_log_csv,
    compute_classification_metrics,
    init_log_csv,
    print_log_header,
    print_log_row,
)

MAX_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

EARLY_STOP_MIN_DELTA = 0.001

SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Chay 1 epoch. Co optimizer -> train; khong co -> eval (no_grad)."""
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
                # Chong exploding gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return total_loss / len(all_labels), np.array(all_labels), np.array(all_preds)


def _is_better(metrics: dict, best_val_acc: float, best_val_f1: float, save_best_by: str) -> bool:
    if save_best_by == "f1":
        return metrics["F1"] > best_val_f1 + EARLY_STOP_MIN_DELTA
    return metrics["Accuracy"] > best_val_acc + EARLY_STOP_MIN_DELTA


def train_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    model_path,
    log_path,
    class_names,
    *,
    model_type: str,
    device,
    dropout: float,
    extra_ckpt: dict | None = None,
    save_best_by: str = "f1",
    early_stop_patience: int = 25,
):
    """
    Huan luyen model + luu checkpoint tot nhat.

    save_best_by: "f1" hoac "accuracy" — tieu chi chon model tot nhat.
    early_stop_patience: so epoch khong cai thien thi dung som.
    extra_ckpt: cac key bo sung luu kem (variant, hidden_size, use_velocity, ...).
    """
    extra_ckpt = extra_ckpt or {}
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    init_log_csv(log_path)
    print_log_header()
    print(
        f"Config: max_epochs={MAX_EPOCHS} | lr={LR} | dropout={dropout} | "
        f"save_best_by={save_best_by} | early_stop={early_stop_patience}",
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
        append_log_csv(log_path, epoch, train_loss, val_loss, metrics)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < prev_lr:
            print(f"  -> Giam LR: {prev_lr:.2e} -> {current_lr:.2e}", flush=True)
            prev_lr = current_lr

        if _is_better(metrics, best_val_acc, best_val_f1, save_best_by):
            best_val_acc = metrics["Accuracy"]
            best_val_f1 = metrics["F1"]
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt = {
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "dropout": dropout,
                "best_val_acc": best_val_acc,
                "best_val_f1": best_val_f1,
                "save_best_by": save_best_by,
                "epoch": epoch,
                "num_params": model.count_parameters(),
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                **extra_ckpt,
            }
            torch.save(ckpt, model_path)
            print(
                f"  -> Luu model tot nhat (val acc={best_val_acc:.4f}, f1={best_val_f1:.4f})",
                flush=True,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(
                f"\nEarly stopping tai epoch {epoch} "
                f"(khong cai thien {save_best_by} trong {early_stop_patience} epoch)",
                flush=True,
            )
            break

    print(
        f"\nBest epoch: {best_epoch} | val acc={best_val_acc:.4f} | val f1={best_val_f1:.4f}",
        flush=True,
    )
    return best_val_acc
