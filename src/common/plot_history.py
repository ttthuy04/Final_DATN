"""
Ve bieu do training history tu training_log.csv.

Usage:
    python src/common/plot_history.py --model lstm
    python src/common/plot_history.py --model stgcn
    python src/common/plot_history.py --model agcn
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import ModelPaths


def load_log(csv_path: Path) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay {csv_path}. Hay train truoc de tao file log."
        )

    epochs, train_loss, val_loss, accuracy, f1 = [], [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["Epoch"]))
            train_loss.append(float(row["Training Loss"]))
            val_loss.append(float(row["Validation Loss"]))
            accuracy.append(float(row["Accuracy"]))
            f1.append(float(row["F1"]))

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": accuracy,
        "f1": f1,
    }


def plot_history(data: dict, title: str, out_path: Path):
    epochs = data["epochs"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(epochs, data["train_loss"], label="Training Loss", color="blue", alpha=0.7)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, data["val_loss"], label="Validation Loss", color="red", alpha=0.7)
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, data["accuracy"], label="Validation Accuracy", color="green", alpha=0.7)
    axes[1, 0].set_title("Validation Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, data["f1"], label="Validation F1", color="purple", alpha=0.7)
    axes[1, 1].set_title("Validation F1 Score")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Ve training history tu CSV log")
    parser.add_argument(
        "--model",
        choices=["lstm", "stgcn", "agcn"],
        default="lstm",
        help="Mo hinh can ve (mac dinh: lstm)",
    )
    args = parser.parse_args()

    paths = ModelPaths(args.model)
    title = f"{args.model.upper()} Training History"
    data = load_log(paths.training_log)
    plot_history(data, title, paths.training_history)
    print(f"Da luu bieu do: {paths.training_history}")


if __name__ == "__main__":
    main()
