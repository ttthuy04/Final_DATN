"""Evaluate LSTM — chay: python src/lstm/evaluate.py"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.evaluate_utils import save_confusion_matrix, save_test_metrics
from src.common.paths import ModelPaths
from src.dataset.skeleton_dataset import get_dataloaders
from src.lstm.model import SkeletonLSTM

PATHS = ModelPaths("lstm")


def load_model(device):
    if not PATHS.model_path.exists():
        raise FileNotFoundError(
            f"Chua co model tai {PATHS.model_path}. Hay chay: python src/lstm/train.py"
        )
    ckpt = torch.load(PATHS.model_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = SkeletonLSTM(
        num_classes=len(class_names),
        hidden_size=ckpt.get("hidden_size", 128),
        num_layers=ckpt.get("num_layers", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, ckpt


@torch.no_grad()
def predict_loader(model, loader, device):
    all_preds, all_labels = [], []
    for X, y in loader:
        X = X.to(device)
        preds = model(X).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Model: LSTM")

    _, _, test_loader, _ = get_dataloaders()
    model, class_names, ckpt = load_model(device)
    print(f"Model: LSTM ({ckpt.get('variant', 'default')})")
    print(f"Parameters: {ckpt.get('num_params', sum(p.numel() for p in model.parameters()))}")
    if ckpt.get("class_weights"):
        print(f"Class weights: {dict(zip(class_names, ckpt['class_weights']))}")

    y_true, y_pred = predict_loader(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n=== LSTM — KET QUA TEST SET ===")
    print(f"So mau test: {len(y_true)}")
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall    (macro): {rec:.4f}")
    print(f"F1        (macro): {f1:.4f}")
    print(f"Best val acc khi train: {ckpt.get('best_val_acc', 'N/A')}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    metrics = {
        "model": "lstm",
        "variant": ckpt.get("variant", "default"),
        "class_weights": ckpt.get("class_weights"),
        "num_params": int(ckpt.get("num_params", sum(p.numel() for p in model.parameters()))),
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "test_samples": int(len(y_true)),
        "class_names": class_names,
        "best_val_acc": float(ckpt.get("best_val_acc", 0)),
        "best_val_f1": float(ckpt.get("best_val_f1", 0)),
    }
    save_test_metrics(metrics, PATHS.test_metrics)
    save_confusion_matrix(
        y_true, y_pred, class_names, PATHS.confusion_matrix, title="LSTM Confusion Matrix (Test set)"
    )
    print(f"\nDa luu metrics: {PATHS.test_metrics}")
    print(f"Da luu confusion matrix: {PATHS.confusion_matrix}")


if __name__ == "__main__":
    main()
