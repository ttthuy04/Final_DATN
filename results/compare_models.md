# So sanh LSTM vs ST-GCN vs 2s-AGCN

Dataset: 4000 clip skeleton (60 frame, 25 khop), 4 lop hanh vi, split 70/15/15.
Test set: 601 mau. Cung class weights: normal=1.6, sleep=1.0, standing=1.0, use_phone=1.3.

## 1. Bang so sanh tong the (Test set)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Tham so | Best val acc |
|-------|----------|-------------------|----------------|------------|---------|--------------|
| **LSTM** | 86.69% | 0.8696 | 0.8668 | 0.8669 | 631,812 | 88.50% |
| **ST-GCN** | 87.52% | 0.8783 | 0.8751 | 0.8755 | 1,991,094 | 89.67% |
| **2s-AGCN** | 84.86% | 0.8489 | 0.8485 | 0.8470 | 3,585,232 | 85.67% |

**Ket luan tong the:** ST-GCN dat accuracy cao nhat tren test set.

## 2. Bang chi tiet tung lop

| Lop | LSTM F1 | ST-GCN F1 | 2s-AGCN F1 |
|-----|--------|--------|--------|
| normal | 0.81 | 0.84 | 0.80 |
| sleep | 0.89 | 0.86 | 0.88 |
| standing | 0.95 | 0.96 | 0.92 |
| use_phone | 0.81 | 0.83 | 0.79 |

## 3. Giai thich mo hinh

### BiLSTM
- **Mo ta:** Mang LSTM 2 chieu doc chuoi skeleton (60 frame x 100 dac trung). Hoc phu thuoc thoi gian doc lap tren tung frame.
- **Ket qua test:** Accuracy 86.69%, F1 macro 0.8669.

### ST-GCN (Yan et al., AAAI 2018)
- **Mo ta:** Graph Convolution + Temporal Convolution tren 25 khop upper-body. Spatial configuration partitioning + learnable edge importance.
- **Ket qua test:** Accuracy 87.52%, F1 macro 0.8755.

### 2s-AGCN (Shi et al., CVPR 2019)
- **Mo ta:** Hai stream Joint + Bone voi Adaptive Graph Convolution. Hoc quan he khop dong + huong xuong.
- **Ket qua test:** Accuracy 84.86%, F1 macro 0.8470.

## 4. De xuat su dung

- **Nhan dien chinh xac nhat:** dung **ST-GCN**.
- **Can toc do / tai nguyen thap:** xem xet LSTM.
