import torch
import torch.nn as nn


class SkeletonLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 100,
        hidden_size: int = 128, #số chiều trạng thía ẩn lstm
        num_layers: int = 2, #số lớp lstm
        num_classes: int = 4, # phân loại 4 hành vi
        dropout: float = 0.3, #0.3 dropout de tranh overfitting
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, # bidirectional=True để lstm có thể học theo 2 chiều , nhờ đó mô hình học dc thông tin trc và sau 
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes) # fc là fully connected layer , để chuyển đổi từ hidden size sang num classes

    def forward(self, x):
        out, _ = self.lstm(x) #(32,60,256)
        return self.fc(out[:, -1, :]) #lấy output cuối cùng của frame cuối cùng của lstm
#(32,4) ,32 video trả về 32 dự đoán , mỗi dự đoán có lớp tương ứng
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
