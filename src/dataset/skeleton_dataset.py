import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKELETON_DIR = PROJECT_ROOT / "data" / "skeleton" # đường dẫn đến thư mục skeleton
N_FRAMES = 60 # 60 frame
N_FEATURES = 100 # 100 features

#load dataaset + chia train/val/test
def _discover_classes():
    if not SKELETON_DIR.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc: {SKELETON_DIR}")
    classes = sorted(
        d.name for d in SKELETON_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not classes:
        raise FileNotFoundError(f"Khong co nhan nao trong {SKELETON_DIR}")
    return classes


CLASS_NAMES = _discover_classes() # danh sách các nhãn
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)} # đổi nhãn thành index (0, 1, 2, 3)

#biến dữ liệu thành dạng pytourch có thể train
class SkeletonDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32) # chuyển dữ liệu numpy sang pytourch tensor
        self.y = torch.tensor(y, dtype=torch.long)
# X là dữ liệu skeleton , y là nhãn số
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#gom toàn bộ file numpy thành X (N, 60, 100), y (N,)
def load_skeleton_data(skeleton_dir: Path = SKELETON_DIR):
    """Doc tat ca file .npy -> X (N, 60, 100), y (N,)."""
    X_list, y_list, paths = [], [], []
#X : dl skeleton , y: dl nhãn số, paths: dl đường dẫn file numpy
    for label in CLASS_NAMES:
        label_dir = skeleton_dir / label
        if not label_dir.is_dir():
            continue
        for npy_path in sorted(label_dir.glob("*.npy")):
            data = np.load(npy_path)
            if data.shape != (N_FRAMES, N_FEATURES):
                print(f"[WARN] Bo qua {npy_path.name}: shape {data.shape}")
                continue
            X_list.append(data.astype(np.float32))
            y_list.append(LABEL_TO_IDX[label])
            paths.append(str(npy_path))

    if not X_list:
        raise RuntimeError("Khong load duoc mau nao. Hay chay extract_skeleton.py truoc.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y, paths
#chia dl

def get_dataloaders(
    batch_size: int = 32,
    train_ratio: float = 0.7, #70% train, 15% val, 15% test
    val_ratio: float = 0.15, #15% val, 15% test
    test_ratio: float = 0.15,
    random_state: int = 42, # random state để chia dữ liệu giống nhau
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    X, y, paths = load_skeleton_data()
#chia lần 1
    X_train, X_temp, y_train, y_temp, paths_train, paths_temp = train_test_split(
        X,
        y,
        paths,
        test_size=(1 - train_ratio), # 30% test, 70% train
        stratify=y, 
        random_state=random_state,
    )

    val_share = val_ratio / (val_ratio + test_ratio) # 0,15 / (0,15 + 0,15) = 0,5
    X_val, X_test, y_val, y_test, paths_val, paths_test = train_test_split(
        X_temp,
        y_temp,
        paths_temp,
        test_size=(1 - val_share), # 0,5 test, 0,5 val của 0,3 
        stratify=y_temp,
        random_state=random_state,
    )
#(32,60,100)
    train_loader = DataLoader(
        SkeletonDataset(X_train, y_train), batch_size=batch_size, shuffle=True # shuffle=True để trộn dữ liệu
    )
    val_loader = DataLoader(SkeletonDataset(X_val, y_val), batch_size=batch_size, shuffle=False) # shuffle=False để không trộn dữ liệu
    test_loader = DataLoader(
        SkeletonDataset(X_test, y_test), batch_size=batch_size, shuffle=False # shuffle=False để không trộn dữ liệu
    )
#thông tin chia dữ liệu
    split_info = {
        "total": len(y),
        "train": len(y_train),
        "val": len(y_val),
        "test": len(y_test),
        "class_names": CLASS_NAMES,
        "paths_train": paths_train,
        "paths_val": paths_val,
        "paths_test": paths_test,
    }
    return train_loader, val_loader, test_loader, split_info


if __name__ == "__main__":
    train_loader, val_loader, test_loader, info = get_dataloaders()
    X_sample, y_sample = next(iter(train_loader))
    print("CLASS_NAMES:", info["class_names"])
    print(
        f"Total: {info['total']} | Train: {info['train']} | "
        f"Val: {info['val']} | Test: {info['test']}"
    )
    print(f"Batch X shape: {X_sample.shape} | Batch y shape: {y_sample.shape}")
