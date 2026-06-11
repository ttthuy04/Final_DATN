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
NUM_COORDS = 2
NUM_PARTITIONS = 3  # spatial configuration (Yan et al. 2018)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
SPATIAL_CENTER = 0  # nose lam moc hop graph cho upper-body

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
]


def _hop_distance(num_joints: int, edges, center: int) -> np.ndarray:
    """BFS hop distance tu center tren graph xuong."""
    dist = np.full(num_joints, np.inf, dtype=np.float32)
    dist[center] = 0.0
    adj = [[] for _ in range(num_joints)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    queue = [center]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1.0
                queue.append(v)
    dist[np.isinf(dist)] = 999.0
    return dist


def _normalize_adjacency(A: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    """D^-1/2 (A+I) D^-1/2 theo Eq.9 ST-GCN paper."""
    A = A + np.eye(A.shape[0], dtype=np.float32)
    D = np.sum(A, axis=1)
    D = D + alpha
    inv_sqrt = np.power(D, -0.5, dtype=np.float32)
    D_inv = np.diag(inv_sqrt)
    return D_inv @ A @ D_inv


def build_spatial_configuration(num_joints: int = NUM_JOINTS, edges=None) -> np.ndarray:
    """
    Spatial configuration partitioning (Yan et al., AAAI 2018, Sec 3.4):
    A0 = self, A1 = inward (centripetal), A2 = outward (centrifugal).
    """
    edges = edges or SKELETON_EDGES
    dist = _hop_distance(num_joints, edges, SPATIAL_CENTER)

    A_self = np.eye(num_joints, dtype=np.float32)
    A_inward = np.zeros((num_joints, num_joints), dtype=np.float32)
    A_outward = np.zeros((num_joints, num_joints), dtype=np.float32)

    for i, j in edges:
        if dist[i] == dist[j]:
            A_inward[i, j] = 1.0
            A_inward[j, i] = 1.0
        elif dist[i] < dist[j]:
            A_inward[i, j] = 1.0
            A_outward[j, i] = 1.0
        else:
            A_inward[j, i] = 1.0
            A_outward[i, j] = 1.0

    partitions = np.stack(
        [
            _normalize_adjacency(A_self),
            _normalize_adjacency(A_inward),
            _normalize_adjacency(A_outward),
        ],
        axis=0,
    )
    return partitions.astype(np.float32)


class GraphConv(nn.Module):
    """
    Graph conv voi K partition + learnable edge importance M (Yan et al. 2018).
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.num_partitions = A.shape[0]
        self.register_buffer("A", A)
        self.PA = nn.Parameter(torch.ones_like(A))
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.num_partitions,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        n, kc, t, v = x.shape
        x = x.view(n, self.num_partitions, kc // self.num_partitions, t, v)
        A = self.A * self.PA
        out = 0
        for k in range(self.num_partitions):
            x_k = torch.einsum("vw,nctv->nctw", A[k], x[:, k])
            out = out + x_k
        return out


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
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=(stride, 1)
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class SkeletonSTGCN(nn.Module):
    """forward(x): x (N, 60, 100) -> logits (N, num_classes)"""

    def __init__(
        self,
        num_classes: int = 4,
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

        if use_spatial_partition:
            A_np = build_spatial_configuration(num_joints)
        else:
            # Fallback: uni-labeling (single normalized A+I)
            A_single = np.eye(num_joints, dtype=np.float32)
            for i, j in SKELETON_EDGES:
                A_single[i, j] = 1.0
                A_single[j, i] = 1.0
            A_np = _normalize_adjacency(A_single)[np.newaxis, ...]

        A = torch.tensor(A_np, dtype=torch.float32)
        self.st_blocks = nn.ModuleList(
            [
                STGCNBlock(self.in_channels, 64, A, stride=1, dropout=dropout),
                STGCNBlock(64, 64, A, stride=1, dropout=dropout),
                STGCNBlock(64, 128, A, stride=2, dropout=dropout),
                STGCNBlock(128, 128, A, stride=1, dropout=dropout),
                STGCNBlock(128, 256, A, stride=2, dropout=dropout),
                STGCNBlock(256, 256, A, stride=1, dropout=dropout),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def flat_to_graph(
        x: torch.Tensor,
        normalize_root: bool = False,
        use_velocity: bool = False,
    ) -> torch.Tensor:
        N, T, _ = x.shape
        x = x.view(N, T, NUM_JOINTS, 4)
        coords = x[..., :NUM_COORDS].clone()

        if normalize_root:
            root = (coords[:, :, LEFT_SHOULDER] + coords[:, :, RIGHT_SHOULDER]) / 2.0
            coords = coords - root.unsqueeze(2)

        pos = coords.permute(0, 3, 1, 2).contiguous()

        if use_velocity:
            vel = pos[:, :, 1:, :] - pos[:, :, :-1, :]
            vel = F.pad(vel, (0, 0, 1, 0), mode="constant", value=0.0)
            pos = torch.cat([pos, vel], dim=1)

        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flat_to_graph(
            x,
            normalize_root=self.normalize_root,
            use_velocity=self.use_velocity,
        )
        for block in self.st_blocks:
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
