"""Vong lap train ST-GCN (Yan et al. AAAI 2018 + toi uu hoa)."""

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
DROPOUT = 0.25 # dropout để tránh overfitting , tắt đi 1 phần để tránh học vẹt
GRAD_CLIP = 1.0

EARLY_STOP_PATIENCE = 25 # 25 epoch không cải thiện thì dừng
EARLY_STOP_MIN_DELTA = 0.001

SCHEDULER_PATIENCE = 5 # 5 epoch không cải thiện thì giảm learning rate
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# Luu best theo F1 macro (on dinh hon accuracy khi val dao dong)
SAVE_BEST_BY = "f1"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, device, optimizer=None): #hàm để chạy 1 epoch, train thì optimizer = true, còn val thì flasse
    is_train = optimizer is not None #nếu optimizer không None thì là train mode
    model.train() if is_train else model.eval() #nếu train mode thì train, nếu eval mode thì eval

    total_loss = 0.0 #tổng loss
    all_labels, all_preds = [], [] #lưu nhãn và dự đoán

    ctx = torch.enable_grad() if is_train else torch.no_grad() #nếu train mode thì enable grad, nếu eval mode thì disable grad
    with ctx:
        for X, y in loader: #X là dữ liệu skeleton (32,60,100) , y là nhãn số (32,)
            X, y = X.to(device), y.to(device)
            if is_train: # xóa grandient để tránh tích lũy
                optimizer.zero_grad()
            logits = model(X) #dự đoán
            loss = criterion(logits, y) #tính loss
            if is_train:
                loss.backward() #tính gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP) #clip gradient để tránh overfitting
                optimizer.step() #cập nhật weights

            total_loss += loss.item() * X.size(0) #tính total loss
            preds = logits.argmax(dim=1) #lấy dự đoán có độ lớn nhất
            all_labels.extend(y.cpu().numpy()) #lưu nhãn thajt
            all_preds.extend(preds.cpu().numpy()) #lưu dự đoán 

    return total_loss / len(all_labels), np.array(all_labels), np.array(all_preds) #trả về loss, nhãn và dự đoán

 #hàm để kiểm tra xem model có tốt hơn model cũ không
def _is_better(metrics: dict, best_val_acc: float, best_val_f1: float) -> bool:
    if SAVE_BEST_BY == "f1": #nếu lưu best theo f1
        return metrics["F1"] > best_val_f1 + EARLY_STOP_MIN_DELTA #nếu f1 lớn hơn best_val_f1 thì trả về True
    return metrics["Accuracy"] > best_val_acc + EARLY_STOP_MIN_DELTA #nếu accuracy lớn hơn best_val_acc thì trả về True


def train_stgcn( #hàm để huấn luyện model
    model,
    train_loader, #loader train
    val_loader, #loader val
    class_weights, #trọng số class
    model_path, #đường dẫn lưu model
    log_path, #đường dẫn lưu log
    class_names, #tên class
    extra_ckpt: dict, #thêm checkpoint
    device, #device
):
    class_weights = class_weights.to(device) #đổi thành tensor
    criterion = nn.CrossEntropyLoss(weight=class_weights) #tính loss    
    optimizer = torch.optim.Adam( #optimizer là Adam
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( #scheduler là ReduceLROnPlateau
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    init_log_csv(log_path)
    print_log_header()
    print(
        f"Config: max_epochs={MAX_EPOCHS} | lr={LR} | dropout={DROPOUT} | "
        f"save_best_by={SAVE_BEST_BY} | early_stop={EARLY_STOP_PATIENCE} | "
        f"paper=ST-GCN Yan2018 spatial_partition+edge_importance",
        flush=True,
    )

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    prev_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(1, MAX_EPOCHS + 1): #duyệt qua tất cả các epoch
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer) #chạy epoch train
        val_loss, y_val, y_pred = run_epoch(model, val_loader, criterion, device) #chạy epoch val   
        metrics = compute_classification_metrics(y_val, y_pred) #tính metrics

        print_log_row(epoch, train_loss, val_loss, metrics)
        append_log_csv(log_path, epoch, train_loss, val_loss, metrics) #lưu log

        scheduler.step(val_loss) #giảm learning rate nếu val loss không giảm
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < prev_lr:
            print(f"  -> Giam LR: {prev_lr:.2e} -> {current_lr:.2e}", flush=True)
            prev_lr = current_lr

        if _is_better(metrics, best_val_acc, best_val_f1): #nếu model tốt hơn model cũ thì lưu best
            best_val_acc = metrics["Accuracy"] #lưu best val acc
            best_val_f1 = metrics["F1"] #lưu best val f1
            best_epoch = epoch #lưu best epoch
            epochs_no_improve = 0 #reset epochs no improve
            ckpt = {
                "model_type": "stgcn", #model type
                "model_state_dict": model.state_dict(), #model state dict
                "class_names": class_names, #tên class
                "dropout": DROPOUT, #dropout
                "best_val_acc": best_val_acc, #best val acc
                "best_val_f1": best_val_f1, #best val f1
                "save_best_by": SAVE_BEST_BY, #save best by
                "epoch": epoch, #epoch
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

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping tai epoch {epoch} "
                f"(khong cai thien {SAVE_BEST_BY} trong {EARLY_STOP_PATIENCE} epoch)",
                flush=True,
            )
            break

    print(
        f"\nBest epoch: {best_epoch} | val acc={best_val_acc:.4f} | val f1={best_val_f1:.4f}",
        flush=True,
    )
    return best_val_acc
