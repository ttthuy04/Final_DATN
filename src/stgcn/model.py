"""
ST-GCN (Yan et al., AAAI 2018) cho skeleton upper-body 25 khop.

Cai tien theo bai bao goc:
- Spatial configuration partitioning (3 subgraph: root / inward / outward)
- Learnable edge importance mask M (Section 3.5)
- ST-GCN block: GCN + temporal TCN kernel=9

Bo sung pho bien trong skeleton action recognition:
- Joint velocity stream (Peng et al., similar pipelines)
- Root normalization theo vai (landmark 11, 12)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_JOINTS = 25 #25 joints
NUM_COORDS = 2 # 2D coordinates
NUM_PARTITIONS = 3  # spatial configuration (Yan et al. 2018) học chính nó, từ xa đến gần trung tâm , và ngược lại
LEFT_SHOULDER = 11  # đỉnh bên trái của vai , lấy làm trung tâm
RIGHT_SHOULDER = 12  # đỉnh bên phải của vai , lấy làm trung tâm
SPATIAL_CENTER = 0  # nose lam moc hop graph cho upper-body , lấy làm trung tâm

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
]

# tính khoảng cách từ node trung tâm đến node khác
def _hop_distance(num_joints: int, edges, center: int) -> np.ndarray:
    """BFS hop distance tu center tren graph xuong."""
    dist = np.full(num_joints, np.inf, dtype=np.float32)
    dist[center] = 0.0 #nút trung tâm = 0
    adj = [[] for _ in range(num_joints)] # tạo danh sách kề cho mỗi nút
    for i, j in edges:
        adj[i].append(j) # thêm cạnh từ i đến j
        adj[j].append(i) # thêm cạnh từ j đến i , vì skeleton là graph 2 chiều 
# bắt đầu tìm kiếm theo chiều rộng 
    queue = [center]
    while queue:
        u = queue.pop(0) # lấy nút đầu tiên trong queue
        for v in adj[u]: # duyệt qua tất cả các nút kề với u
            if dist[v] == np.inf: # nếu nút v chưa được duyệt
                dist[v] = dist[u] + 1.0 # cập nhật trạng thái
                queue.append(v)
    dist[np.isinf(dist)] = 999.0 # nếu nút v chưa được duyệt thì đặt giá trị lớn nhất
    return dist

# chuẩn hóa ma trận kề => scale ma trận theo tính toán hợp lý, ổn định hơn
def _normalize_adjacency(A: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """D^-1/2 (A+I) D^-1/2 theo Eq.9 ST-GCN paper."""
    A = A + np.eye(A.shape[0], dtype=np.float32) # mỗi nút tự kề với chính nó
    D = np.sum(A, axis=1)
    D = D + alpha # để tránh chia cho 0
    inv_sqrt = np.power(D, -0.5, dtype=np.float32)
    D_inv = np.diag(inv_sqrt) # tạo ma trận đường chéo từ inv_sqrt
    return D_inv @ A @ D_inv # D^-1/2 (A) D^-1/2

# tạo 3 ma trận kề
def build_spatial_configuration(num_joints: int = NUM_JOINTS, edges=None) -> np.ndarray:
    """
    Spatial configuration partitioning (Yan et al., AAAI 2018, Sec 3.4):
    A0 = self, A1 = inward (centripetal), A2 = outward (centrifugal).
    """
    edges = edges or SKELETON_EDGES
    dist = _hop_distance(num_joints, edges, SPATIAL_CENTER) # tính toán hop

    A_self = np.eye(num_joints, dtype=np.float32) # ma trận đơn vị A0
    A_inward = np.zeros((num_joints, num_joints), dtype=np.float32) # ma trận không A1
    A_outward = np.zeros((num_joints, num_joints), dtype=np.float32) # ma trận không A2

    for i, j in edges: 
        if dist[i] == dist[j]: # nếu 2 nút có cùng khoảng cách từ trung tâm
            A_inward[i, j] = 1.0
            A_inward[j, i] = 1.0
        elif dist[i] < dist[j]: # nếu nút i gần trung tâm hơn nút j
            A_inward[i, j] = 1.0
            A_outward[j, i] = 1.0
        else: # nếu nút j gần trung tâm hơn nút i
            A_inward[j, i] = 1.0
            A_outward[i, j] = 1.0

    partitions = np.stack( # stack 3 ma trận thành 1 ma trận 3x25x25
        [
            _normalize_adjacency(A_self),
            _normalize_adjacency(A_inward),
            _normalize_adjacency(A_outward),
        ],
        axis=0,
    )
    return partitions.astype(np.float32)  

#GCN : học quan hệ giữa các khớp trong cùng 1 frame
class GraphConv(nn.Module):
    """
    Graph conv voi K partition + learnable edge importance M (Yan et al. 2018).
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.num_partitions = A.shape[0] # 3, self, inward, outward
        self.register_buffer("A", A)
        self.PA = nn.Parameter(torch.ones_like(A)) # (3,25,25)
        self.conv = nn.Conv2d(  
            in_channels, # 4 x, ,y, vx, ,vy
            out_channels * self.num_partitions, # 64 x3 
            kernel_size=1,
        ) #(N,3,64,60,25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x) # (N,64x3,T,V) conv 1x1
        n, kc, t, v = x.shape # 32, 192, 64, 25
        x = x.view(n, self.num_partitions, kc // self.num_partitions, t, v) # TÁCH THÀNH 3 MA TRẬN (N,3,64,T,V)
        A = self.A * self.PA # (3,25,25) * (3,25,25) = (3,25,25)  , tự học quan hệ giữa các khớp trong cùng 1 frame
        out = 0
        for k in range(self.num_partitions): #duyệt qua 3 partition
            x_k = torch.einsum("vw,nctv->nctw", A[k], x[:, k]) # 25,25 * (N,64,T,25) = (N,64,T,25) #GCN
            out = out + x_k
        return out # (N,64,T,V)  gcn 


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A) # gcn : học quan hệ giữa các khớp
        self.tcn = nn.Sequential( # tcn: hocj quan hệ giauwx các frame
            nn.BatchNorm2d(out_channels), # batch norm : chuẩn hóa dữ liệu
            nn.ReLU(inplace=True), # relu : hàm kích hoạt phi tuyến tính
            nn.Conv2d(  # tcn
                out_channels,
                out_channels,
                kernel_size=(9, 1), # 9 frame theo gcn , 1 frame theo tcn
                stride=(stride, 1), # stride: bước nhảy, nếu stride =1 thì giữ nguyên số frame, nếu stride =2 thì giảm số frame đi 1 nửa
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels), # batch norm : chuẩn hóa dữ liệu
            nn.Dropout(dropout, inplace=True), # dropout : tránh overfitting
        ) 
        if in_channels != out_channels or stride != 1: # nếu input  khác output , thì càn chỉnh lại inout
            self.residual = nn.Conv2d( # conv 1x1 để chỉnh lại input và output
                in_channels, out_channels, kernel_size=1, stride=(stride, 1)
            )
        else:
            self.residual = nn.Identity() # identity : để giữ nguyên input và output
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x)) #output = relu(tcn(gcn(x)) + residual(x)) , học quan hệ giữa các khớp trong cùng 1 frame và học quan hệ giữa các frame liên tiếp


class SkeletonSTGCN(nn.Module):
    """forward(x): x (N, 60, 100) -> logits (N, num_classes)""" # (32,60,100) --> (32,4)

    def __init__(
        self,
        num_classes: int = 4, # 4 hành vi
        num_joints: int = NUM_JOINTS,
        dropout: float = 0.25,
        use_velocity: bool = False,
        normalize_root: bool = False,
        use_spatial_partition: bool = True,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.use_velocity = use_velocity
        self.normalize_root = normalize_root
        self.use_spatial_partition = use_spatial_partition
        self.in_channels = NUM_COORDS * (2 if use_velocity else 1)

        if use_spatial_partition: # nếu sử dụng spatial configuration partitioning
            A_np = build_spatial_configuration(num_joints) # tạo 3 ma trận kề cho 25 khớp
        else: # nếu không sử dụng spatial configuration partitioning
            # Fallback: uni-labeling (single normalized A+I)
            A_single = np.eye(num_joints, dtype=np.float32) # ma trận đơn vị
            for i, j in SKELETON_EDGES: # duyệt qua tất cả các cạnh trong skeleton
                A_single[i, j] = 1.0 # thêm cạnh từ i đến j
                A_single[j, i] = 1.0 # thêm cạnh từ j đến i
            A_np = _normalize_adjacency(A_single)[np.newaxis, ...] # chuẩn hóa ma trận kề

        A = torch.tensor(A_np, dtype=torch.float32) # đổi numpy thành tensor
        self.st_blocks = nn.ModuleList( # tạo 6 block stgcn
            [
                STGCNBlock(self.in_channels, 64, A, stride=1, dropout=dropout),
                STGCNBlock(64, 64, A, stride=1, dropout=dropout),
                STGCNBlock(64, 128, A, stride=2, dropout=dropout),
                STGCNBlock(128, 128, A, stride=1, dropout=dropout),
                STGCNBlock(128, 256, A, stride=2, dropout=dropout),
                STGCNBlock(256, 256, A, stride=1, dropout=dropout),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # lấy trung bình theo thời gian (N,256,1,1)
        self.fc = nn.Linear(256, num_classes) #(N,256) --> (N,4)  classification layer , phân loại hành vi

    @staticmethod
    def flat_to_graph( # đổi dữ liệu từ dạng flat sang dạng graph (N,60,25,4)
        x: torch.Tensor,
        normalize_root: bool = False,
        use_velocity: bool = False,
    ) -> torch.Tensor:
        N, T, _ = x.shape #lấy kích thước batch, số frame, số khớp
        x = x.view(N, T, NUM_JOINTS, 4) # (N,60,25,4)
        coords = x[..., :NUM_COORDS].clone() # lấy toạ độ x, y, (N,60,25,2)

        if normalize_root:
            root = (coords[:, :, LEFT_SHOULDER] + coords[:, :, RIGHT_SHOULDER]) / 2.0 # tính toán tọa độ trung tâm của vai
            coords = coords - root.unsqueeze(2) # trừ tọa độ trung tâm của vai khỏi tọa độ các khớp
# giảm ảnh hưởng vị trí sinh viên trong lớp học
        pos = coords.permute(0, 3, 1, 2).contiguous() # đổi thứ tự các trục để phù hợp với input của gcn N,2,60,25
# tính vận tốc giữa các frame liên tiếp
        if use_velocity:
            vel = pos[:, :, 1:, :] - pos[:, :, :-1, :]
            vel = F.pad(vel, (0, 0, 1, 0), mode="constant", value=0.0) 
            pos = torch.cat([pos, vel], dim=1)

        return pos
# ghép tọa độ vận tốc : (N,4,60,25) của posision và velocity


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flat_to_graph(
            x,
            normalize_root=self.normalize_root,
            use_velocity=self.use_velocity,
        )
        for block in self.st_blocks: # cho qua 6 block ở trên
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        return self.fc(x) # classification layer (N,4) , phân loại hành vi 

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# (N,64,60,25)
#(N, 64,60,25)
#(N, 128,30,25)
#(N, 128,30,25)
#(N, 256,15,25)
#(N, 256,15,25)