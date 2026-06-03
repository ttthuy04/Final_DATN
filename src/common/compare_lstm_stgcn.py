"""
So sanh LSTM vs ST-GCN — tao bang + file giai thich.

Chay: python src/common/compare_lstm_stgcn.py
Output:
  results/compare_lstm_stgcn.csv
  results/compare_lstm_stgcn.md
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

RESULTS_DIR = PROJECT_ROOT / "results"
CSV_PATH = RESULTS_DIR / "compare_lstm_stgcn.csv"
MD_PATH = RESULTS_DIR / "compare_lstm_stgcn.md"

MODEL_INFO = {
    "lstm": {
        "ten": "BiLSTM",
        "mo_ta": (
            "Mang LSTM 2 chieu doc chuoi skeleton (60 frame x 100 dac trung). "
            "Hoc phu thuoc thoi gian doc lap tren tung frame."
        ),
        "uu_diem": (
            "Nhe (~632K tham so), train nhanh, on dinh. Phu hop chuoi thoi gian phang."
        ),
        "nhuoc_diem": (
            "Khong mo hinh hoa ro quan he khop xuong khong gian."
        ),
    },
    "stgcn": {
        "ten": "ST-GCN (Yan et al., AAAI 2018)",
        "mo_ta": (
            "Graph Convolution + Temporal Convolution tren 25 khop upper-body. "
            "Spatial configuration partitioning (3 subgraph) + learnable edge importance."
        ),
        "uu_diem": (
            "Khai thac cau truc khop xuong + chuyen dong (velocity). "
            "Phu hop hanh vi co tu the ro (standing, use_phone)."
        ),
        "nhuoc_diem": (
            "Nhieu tham so hon (~2M), train lau hon, val dao dong manh hon."
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
        w.writerow(["Bang 1: So sanh tong the (Test set 601 mau)"])
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
    lstm_o = next(r for r in overall_rows if r["model"] == "LSTM")
    stgcn_o = next(r for r in overall_rows if r["model"] == "ST-GCN")

    lines = [
        "# So sanh LSTM vs ST-GCN",
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
        "| Lop | LSTM P | LSTM R | LSTM F1 | ST-GCN P | ST-GCN R | ST-GCN F1 | Tot hon |",
        "|-----|--------|--------|---------|----------|----------|-----------|---------|",
    ]
    for cls in class_names:
        l = next(r for r in per_class_rows if r["model"] == "LSTM" and r["class"] == cls)
        s = next(r for r in per_class_rows if r["model"] == "ST-GCN" and r["class"] == cls)
        better = "ST-GCN" if s["f1"] > l["f1"] else ("LSTM" if l["f1"] > s["f1"] else "Hoa")
        lines.append(
            f"| {cls} | {l['precision']:.2f} | {l['recall']:.2f} | {l['f1']:.2f} | "
            f"{s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | {better} |"
        )

    lines += [
        "",
        "## 3. Giai thich mo hinh",
        "",
        "### LSTM (Bidirectional LSTM)",
        f"- **Mo ta:** {MODEL_INFO['lstm']['mo_ta']}",
        f"- **Uu diem:** {MODEL_INFO['lstm']['uu_diem']}",
        f"- **Nhuoc diem:** {MODEL_INFO['lstm']['nhuoc_diem']}",
        f"- **Ket qua test:** Accuracy {lstm_o['accuracy']:.2%}, F1 macro {lstm_o['f1']:.4f}.",
        "",
        "### ST-GCN (Spatial Temporal Graph Convolutional Networks)",
        f"- **Mo ta:** {MODEL_INFO['stgcn']['mo_ta']}",
        f"- **Uu diem:** {MODEL_INFO['stgcn']['uu_diem']}",
        f"- **Nhuoc diem:** {MODEL_INFO['stgcn']['nhuoc_diem']}",
        f"- **Ket qua test:** Accuracy {stgcn_o['accuracy']:.2%}, F1 macro {stgcn_o['f1']:.4f}.",
        "",
        "## 4. Nhan xet so sanh",
        "",
    ]

    diff_acc = stgcn_o["accuracy"] - lstm_o["accuracy"]
    if diff_acc > 0:
        lines.append(
            f"- ST-GCN vuot LSTM **{diff_acc*100:.2f} diem** accuracy tren test set."
        )
        lines.append(
            "- ST-GCN khai thac duoc quan he khong gian giua cac khop (graph conv) "
            "va chuyen dong (velocity stream), phu hop nhan dien tu the nhu standing, use_phone."
        )
        lines.append(
            "- LSTM van co loi the ve toc do train va so tham so it hon (~3 lan nhe hon ST-GCN)."
        )
    else:
        lines.append(f"- LSTM vuot ST-GCN {abs(diff_acc)*100:.2f} diem accuracy.")

    # Per-class insights
    for cls in class_names:
        l = next(r for r in per_class_rows if r["model"] == "LSTM" and r["class"] == cls)
        s = next(r for r in per_class_rows if r["model"] == "ST-GCN" and r["class"] == cls)
        if abs(l["f1"] - s["f1"]) >= 0.03:
            better = "ST-GCN" if s["f1"] > l["f1"] else "LSTM"
            lines.append(
                f"- Lop **{cls}**: {better} tot hon (F1 {max(l['f1'], s['f1']):.2f} "
                f"vs {min(l['f1'], s['f1']):.2f})."
            )

    lines += [
        "",
        "## 5. De xuat su dung",
        "",
        f"- **Nhan dien chinh xac nhat:** dung **{winner}** (best.pt trong models_saved/).",
        "- **Can toc do / tai nguyen thap:** xem xet LSTM.",
        "- ST-LSTM se duoc bo sung vao bang so sanh khi train xong.",
        "",
    ]

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    lstm_json = load_json_metrics("lstm")
    stgcn_json = load_json_metrics("stgcn")
    class_names = lstm_json["class_names"]

    _, _, test_loader, _ = get_dataloaders()

    lstm_model, _ = load_lstm(device)
    stgcn_model, _ = load_stgcn(device)

    y_true_l, y_pred_l = predict(lstm_model, test_loader, device)
    y_true_s, y_pred_s = predict(stgcn_model, test_loader, device)

    assert np.array_equal(y_true_l, y_true_s)

    overall_rows = []
    per_class_rows = []
    for label, json_m, y_pred in [
        ("LSTM", lstm_json, y_pred_l),
        ("ST-GCN", stgcn_json, y_pred_s),
    ]:
        overall_rows.append({
            "model": label,
            "accuracy": json_m["accuracy"],
            "precision": json_m["precision_macro"],
            "recall": json_m["recall_macro"],
            "f1": json_m["f1_macro"],
            "params": json_m["num_params"],
            "best_val": json_m["best_val_acc"],
        })
        for row in per_class_table(y_true_l, y_pred, class_names):
            per_class_rows.append({"model": label, **row})

    winner = max(overall_rows, key=lambda r: r["accuracy"])["model"]

    write_csv(overall_rows, per_class_rows)
    write_md(overall_rows, per_class_rows, class_names, winner)

    print("\n=== BANG SO SANH LSTM vs ST-GCN (Test set) ===\n")
    print(f"{'Model':<10} {'Accuracy':>10} {'F1 macro':>10} {'Params':>12}")
    print("-" * 44)
    for r in overall_rows:
        print(f"{r['model']:<10} {r['accuracy']:>10.2%} {r['f1']:>10.4f} {r['params']:>12,}")
    print(f"\n=> Tot nhat: {winner}")
    print(f"\nDa luu: {CSV_PATH}")
    print(f"Da luu: {MD_PATH}")


if __name__ == "__main__":
    main()
