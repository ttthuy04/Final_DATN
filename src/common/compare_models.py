"""
So sanh LSTM vs ST-GCN vs 2s-AGCN — tao bang + file giai thich.

Chay: python src/common/compare_models.py
Output:
  results/compare_models.csv
  results/compare_models.md
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import ModelPaths
from src.dataset.skeleton_dataset import get_dataloaders
from src.lstm.model import SkeletonLSTM
from src.stgcn.model import SkeletonSTGCN
from src.agcn.model import SkeletonTwoStreamAGCN

RESULTS_DIR = PROJECT_ROOT / "results"
CSV_PATH = RESULTS_DIR / "compare_models.csv"
MD_PATH = RESULTS_DIR / "compare_models.md"

MODEL_INFO = {
    "lstm": {
        "label": "LSTM",
        "ten": "BiLSTM",
        "mo_ta": (
            "Mang LSTM 2 chieu doc chuoi skeleton (60 frame x 100 dac trung). "
            "Hoc phu thuoc thoi gian doc lap tren tung frame."
        ),
    },
    "stgcn": {
        "label": "ST-GCN",
        "ten": "ST-GCN (Yan et al., AAAI 2018)",
        "mo_ta": (
            "Graph Convolution + Temporal Convolution tren 25 khop upper-body. "
            "Spatial configuration partitioning + learnable edge importance."
        ),
    },
    "agcn": {
        "label": "2s-AGCN",
        "ten": "2s-AGCN (Shi et al., CVPR 2019)",
        "mo_ta": (
            "Hai stream Joint + Bone voi Adaptive Graph Convolution. "
            "Hoc quan he khop dong + huong xuong."
        ),
    },
}


def load_json_metrics(name: str) -> dict:
    path = ModelPaths(name).test_metrics
    if not path.exists():
        raise FileNotFoundError(f"Chua co {path}. Hay chay evaluate truoc.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_lstm(device):
    path = ModelPaths("lstm").model_path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SkeletonLSTM(
        num_classes=len(ckpt["class_names"]),
        hidden_size=ckpt.get("hidden_size", 128),
        num_layers=ckpt.get("num_layers", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def load_stgcn(device):
    path = ModelPaths("stgcn").model_path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SkeletonSTGCN(
        num_classes=len(ckpt["class_names"]),
        dropout=ckpt.get("dropout", 0.25),
        use_velocity=ckpt.get("use_velocity", True),
        normalize_root=ckpt.get("normalize_root", True),
        use_spatial_partition=ckpt.get("use_spatial_partition", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def load_agcn(device):
    path = ModelPaths("agcn").model_path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SkeletonTwoStreamAGCN(
        num_classes=len(ckpt["class_names"]),
        dropout=ckpt.get("dropout", 0.25),
        use_velocity=ckpt.get("use_velocity", True),
        normalize_root=ckpt.get("normalize_root", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict(model, loader, device):
    preds, labels = [], []
    for X, y in loader:
        X = X.to(device)
        pred = model(X).argmax(dim=1).cpu().numpy()
        preds.extend(pred)
        labels.extend(y.numpy())
    return np.array(labels), np.array(preds)


def per_class_table(y_true, y_pred, class_names):
    rows = []
    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        p_bin = (y_pred == i).astype(int)
        tp = np.sum((y_bin == 1) & (p_bin == 1))
        fp = np.sum((y_bin == 0) & (p_bin == 1))
        fn = np.sum((y_bin == 1) & (p_bin == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({"class": name, "precision": prec, "recall": rec, "f1": f1})
    return rows


def write_csv(overall_rows, per_class_rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Bang 1: So sanh tong the (Test set)"])
        w.writerow(["Model", "Accuracy", "Precision macro", "Recall macro", "F1 macro", "Params", "Best val acc"])
        for r in overall_rows:
            w.writerow([
                r["model"], f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
                f"{r['recall']:.4f}", f"{r['f1']:.4f}", r["params"], f"{r['best_val']:.4f}",
            ])
        w.writerow([])
        w.writerow(["Bang 2: Chi tiet tung lop"])
        w.writerow(["Model", "Class", "Precision", "Recall", "F1"])
        for r in per_class_rows:
            w.writerow([
                r["model"], r["class"], f"{r['precision']:.4f}",
                f"{r['recall']:.4f}", f"{r['f1']:.4f}",
            ])


def write_md(overall_rows, per_class_rows, class_names, winner):
    lines = [
        "# So sanh LSTM vs ST-GCN vs 2s-AGCN",
        "",
        "Dataset: 4000 clip skeleton (60 frame, 25 khop), 4 lop hanh vi, split 70/15/15.",
        "Test set: 601 mau. Cung class weights: normal=1.6, sleep=1.0, standing=1.0, use_phone=1.3.",
        "",
        "## 1. Bang so sanh tong the (Test set)",
        "",
        "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Tham so | Best val acc |",
        "|-------|----------|-------------------|----------------|------------|---------|--------------|",
    ]
    for r in overall_rows:
        lines.append(
            f"| **{r['model']}** | {r['accuracy']:.2%} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} | {r['params']:,} | {r['best_val']:.2%} |"
        )

    lines += [
        "",
        f"**Ket luan tong the:** {winner} dat accuracy cao nhat tren test set.",
        "",
        "## 2. Bang chi tiet tung lop",
        "",
    ]

    header = "| Lop |"
    for r in overall_rows:
        header += f" {r['model']} F1 |"
    lines.append(header)
    sep = "|-----|"
    for _ in overall_rows:
        sep += "--------|"
    lines.append(sep)

    for cls in class_names:
        row = f"| {cls} |"
        f1_vals = []
        for model_label in [r["model"] for r in overall_rows]:
            pc = next(
                x for x in per_class_rows
                if x["model"] == model_label and x["class"] == cls
            )
            row += f" {pc['f1']:.2f} |"
            f1_vals.append((model_label, pc["f1"]))
        best = max(f1_vals, key=lambda x: x[1])[0]
        lines.append(row)

    lines += ["", "## 3. Giai thich mo hinh", ""]
    for key in ("lstm", "stgcn", "agcn"):
        info = MODEL_INFO[key]
        o = next(r for r in overall_rows if r["model"] == info["label"])
        lines += [
            f"### {info['ten']}",
            f"- **Mo ta:** {info['mo_ta']}",
            f"- **Ket qua test:** Accuracy {o['accuracy']:.2%}, F1 macro {o['f1']:.4f}.",
            "",
        ]

    lines += [
        "## 4. De xuat su dung",
        "",
        f"- **Nhan dien chinh xac nhat:** dung **{winner}**.",
        "- **Can toc do / tai nguyen thap:** xem xet LSTM.",
        "",
    ]

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    metrics_by_key = {
        "lstm": load_json_metrics("lstm"),
        "stgcn": load_json_metrics("stgcn"),
        "agcn": load_json_metrics("agcn"),
    }
    class_names = metrics_by_key["lstm"]["class_names"]

    _, _, test_loader, _ = get_dataloaders()

    loaders = {
        "lstm": load_lstm(device),
        "stgcn": load_stgcn(device),
        "agcn": load_agcn(device),
    }

    y_true_ref = None
    overall_rows = []
    per_class_rows = []

    for key in ("lstm", "stgcn", "agcn"):
        model, _ = loaders[key]
        json_m = metrics_by_key[key]
        label = MODEL_INFO[key]["label"]
        y_true, y_pred = predict(model, test_loader, device)
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            assert np.array_equal(y_true_ref, y_true)

        overall_rows.append({
            "model": label,
            "accuracy": json_m["accuracy"],
            "precision": json_m["precision_macro"],
            "recall": json_m["recall_macro"],
            "f1": json_m["f1_macro"],
            "params": json_m["num_params"],
            "best_val": json_m["best_val_acc"],
        })
        for row in per_class_table(y_true, y_pred, class_names):
            per_class_rows.append({"model": label, **row})

    winner = max(overall_rows, key=lambda r: r["accuracy"])["model"]

    write_csv(overall_rows, per_class_rows)
    write_md(overall_rows, per_class_rows, class_names, winner)

    print("\n=== BANG SO SANH 3 MODEL (Test set) ===\n")
    print(f"{'Model':<12} {'Accuracy':>10} {'F1 macro':>10} {'Params':>12}")
    print("-" * 46)
    for r in overall_rows:
        print(f"{r['model']:<12} {r['accuracy']:>10.2%} {r['f1']:>10.4f} {r['params']:>12,}")
    print(f"\n=> Tot nhat: {winner}")
    print(f"\nDa luu: {CSV_PATH}")
    print(f"Da luu: {MD_PATH}")


if __name__ == "__main__":
    main()
