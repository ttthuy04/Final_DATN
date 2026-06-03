"""
2s-AGCN (Shi et al., CVPR 2019) cho skeleton upper-body 25 khop.

- Two-stream: Joint + Bone
- Adaptive graph convolution: A_ad = softmax(A_base + PA)
- ST-GCN block: AGCN + temporal TCN kernel=9
- Root normalization + velocity (giong ST-GCN)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_JOINTS = 25
NUM_COORDS = 2
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
]

# Cay khop huong tu mui (0) — dung cho bone stream
PARENT = {
    0: 0,
    1: 0, 2: 1, 3: 2, 7: 3,
    4: 0, 5: 4, 6: 5, 8: 6,
    9: 0, 10: 0,
    11: 0, 12: 0,
    13: 11, 14: 12,
    15: 13, 16: 14,
    17: 15, 18: 17, 19: 15, 20: 19, 21: 15, 22: 21,
    23: 11, 24: 12,
}


def _normalize_adjacency(A: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    A = A + np.eye(A.shape[0], dtype=np.float32)
    D = np.sum(A, axis=1)
    D = D + alpha
    inv_sqrt = np.power(D, -0.5, dtype=np.float32)
    D_inv = np.diag(inv_sqrt)
    return D_inv @ A @ D_inv


def build_base_adjacency(num_joints: int = NUM_JOINTS, edges=None) -> np.ndarray:
    edges = edges or SKELETON_EDGES
    A = np.eye(num_joints, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return _normalize_adjacency(A)


class AdaptiveGraphConv(nn.Module):
    """Adaptive graph conv: A_ad = softmax(A_base + PA) — Shi et al. 2019."""

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A_base", A)
        self.PA = nn.Parameter(torch.zeros_like(A))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = F.softmax(self.A_base + self.PA, dim=-1)
        x = self.conv(x)
        return torch.einsum("nctv,vw->nctw", x, A)


class AGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, A)
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


class StreamBackbone(nn.Module):
    """Mot stream (joint hoac bone) voi day AGCN blocks."""

    def __init__(
        self,
        in_channels: int,
        A: torch.Tensor,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AGCNBlock(in_channels, 64, A, stride=1, dropout=dropout),
                AGCNBlock(64, 64, A, stride=1, dropout=dropout),
                AGCNBlock(64, 128, A, stride=2, dropout=dropout),
                AGCNBlock(128, 128, A, stride=1, dropout=dropout),
                AGCNBlock(128, 256, A, stride=2, dropout=dropout),
                AGCNBlock(256, 256, A, stride=1, dropout=dropout),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SkeletonTwoStreamAGCN(nn.Module):
    """
    2s-AGCN: Joint stream + Bone stream, fusion bang average logits.
    forward(x): x (N, 60, 100) -> logits (N, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_joints: int = NUM_JOINTS,
        dropout: float = 0.25,
        use_velocity: bool = True,
        normalize_root: bool = True,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.use_velocity = use_velocity
        self.normalize_root = normalize_root
        self.in_channels = NUM_COORDS * (2 if use_velocity else 1)

        A_np = build_base_adjacency(num_joints)
        A = torch.tensor(A_np, dtype=torch.float32)

        self.joint_stream = StreamBackbone(self.in_channels, A, dropout=dropout)
        self.bone_stream = StreamBackbone(self.in_channels, A, dropout=dropout)
        self.fc = nn.Linear(256, num_classes)

        parent_idx = torch.tensor(
            [PARENT[i] for i in range(num_joints)], dtype=torch.long
        )
        self.register_buffer("parent_idx", parent_idx)

    @staticmethod
    def _flat_to_coords(x: torch.Tensor, normalize_root: bool) -> torch.Tensor:
        N, T, _ = x.shape
        x = x.view(N, T, NUM_JOINTS, 4)
        coords = x[..., :NUM_COORDS].clone()

        if normalize_root:
            root = (coords[:, :, LEFT_SHOULDER] + coords[:, :, RIGHT_SHOULDER]) / 2.0
            coords = coords - root.unsqueeze(2)

        return coords

    def _coords_to_graph(self, coords: torch.Tensor) -> torch.Tensor:
        pos = coords.permute(0, 3, 1, 2).contiguous()
        if not self.use_velocity:
            return pos

        vel = pos[:, :, 1:, :] - pos[:, :, :-1, :]
        vel = F.pad(vel, (0, 0, 1, 0), mode="constant", value=0.0)
        return torch.cat([pos, vel], dim=1)

    def _joint_to_bone(self, coords: torch.Tensor) -> torch.Tensor:
        N, T, V, C = coords.shape
        parent = self.parent_idx
        bone = torch.zeros_like(coords)
        for j in range(V):
            p = parent[j].item()
            if p != j:
                bone[:, :, j, :] = coords[:, :, j, :] - coords[:, :, p, :]
        return bone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = self._flat_to_coords(x, self.normalize_root)
        joint_graph = self._coords_to_graph(coords)
        bone_graph = self._coords_to_graph(self._joint_to_bone(coords))

        joint_feat = self.joint_stream(joint_graph)
        bone_feat = self.bone_stream(bone_graph)
        fused = (joint_feat + bone_feat) / 2.0
        return self.fc(fused)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
