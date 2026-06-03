"""Ghi log train: Epoch | Train Loss | Val Loss | Acc | P | R | F1."""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

LOG_COLUMNS = [
    "Epoch",
    "Training Loss",
    "Validation Loss",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
]


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_log_header():
    print(
        f"{'Epoch':>5} | {'Training Loss':>14} | {'Validation Loss':>16} | "
        f"{'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}",
        flush=True,
    )
    print("-" * 95, flush=True)


def print_log_row(epoch: int, train_loss: float, val_loss: float, metrics: dict):
    print(
        f"{epoch:5d} | {train_loss:14.6f} | {val_loss:16.6f} | "
        f"{metrics['Accuracy']:10.6f} | {metrics['Precision']:10.6f} | "
        f"{metrics['Recall']:10.6f} | {metrics['F1']:10.6f}",
        flush=True,
    )


def init_log_csv(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()


def append_log_csv(
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: dict,
):
    row = {
        "Epoch": epoch,
        "Training Loss": round(train_loss, 6),
        "Validation Loss": round(val_loss, 6),
        "Accuracy": round(metrics["Accuracy"], 6),
        "Precision": round(metrics["Precision"], 6),
        "Recall": round(metrics["Recall"], 6),
        "F1": round(metrics["F1"], 6),
    }
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)
