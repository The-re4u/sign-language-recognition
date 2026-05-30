# coding:utf-8
"""Graph Convolutional Network for 21-point hand skeleton topology encoding."""
import numpy as np

# Hand skeleton connectivity from MediaPipe HAND_CONNECTIONS
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Finger bases
]

_FUNCTIONAL_EDGES = [
    (4, 8), (8, 12), (12, 16), (16, 20),  # Fingertip connections
]

# Hand-optimized topology (HA-GCN [9] + Cross Attentive [24]):
#   - Fingertip-to-fingertip full mesh (captures hand "silhouette")
#   - Intra-finger skip: MCP↔DIP (captures overall finger bend independent of PIP)
#   - Symmetric inter-finger MCP cross-links (captures finger spread structure)
_HAND_OPTIMIZED_EXTRA = [
    # Fingertip full mesh (thumb→index→middle→ring→pinky)
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (8, 16), (8, 20),
    (12, 16), (12, 20),
    (16, 20),
    # Intra-finger skip: MCP→DIP (Thumb MCP→IP already in base; IP→TIP here)
    (2, 4), (5, 7), (9, 11), (13, 15), (17, 19),
    # Inter-finger MCP→MCP cross-links (symmetric pairs)
    (5, 9), (5, 13), (5, 17),
    (9, 13), (9, 17),
    (13, 17),
]

# Pre-compute edge index for PyTorch Geometric-style message passing
_BONE_EDGES = _HAND_CONNECTIONS
_ALL_EDGES = _HAND_CONNECTIONS + _FUNCTIONAL_EDGES
_EDGE_INDEX = list(zip(*_ALL_EDGES))
_EDGE_INDEX = (
    list(_EDGE_INDEX[0]) + list(_EDGE_INDEX[1]),
    list(_EDGE_INDEX[1]) + list(_EDGE_INDEX[0])
)

# Hand-optimized edge index (base + functional + optimized extra)
_ALL_OPTIMIZED = _HAND_CONNECTIONS + _FUNCTIONAL_EDGES + _HAND_OPTIMIZED_EXTRA
_EDGE_INDEX_OPTIMIZED = list(zip(*_ALL_OPTIMIZED))
_EDGE_INDEX_OPTIMIZED = (
    list(_EDGE_INDEX_OPTIMIZED[0]) + list(_EDGE_INDEX_OPTIMIZED[1]),
    list(_EDGE_INDEX_OPTIMIZED[1]) + list(_EDGE_INDEX_OPTIMIZED[0])
)


class HandSkeletonGraph:
    """Defines the hand skeleton graph structure."""
    NUM_NODES = 21
    BONE_EDGES = _BONE_EDGES
    ALL_EDGES = _ALL_EDGES
    EDGE_INDEX = _EDGE_INDEX
    EDGE_INDEX_OPTIMIZED = _EDGE_INDEX_OPTIMIZED


def get_edge_index(optimized=True):
    """Return (edge_from, edge_to) tensors for the specified topology."""
    ei = HandSkeletonGraph.EDGE_INDEX_OPTIMIZED if optimized else HandSkeletonGraph.EDGE_INDEX
    return (
        torch.tensor(ei[0], dtype=torch.long),
        torch.tensor(ei[1], dtype=torch.long))


# ============================================================
# Numpy fallback spatial feature extraction (no torch needed)
# ============================================================

def normalize_keypoints(keypoints):
    """Normalize keypoints: center at wrist (index 0), scale by palm size.

    Strategy: first-frame wrist-centric normalization.
      kp' = (kp - wrist) / ||middle_mcp - wrist||

    Eliminates camera distance bias — hands at 30cm and 80cm produce
    identical normalized coordinates. Middle finger MCP (landmark 9) is
    the scale reference (lowest MediaPipe tracking error among MCP joints:
    Hamaguchi 2024 reports r=0.78 for MCP vs r=0.72 for PIP).

    Literature: wrist-anchored normalization improves gesture recognition
    accuracy from 80.56% to 84.66% in comparable systems (2025).
    """
    wrist = keypoints[0].copy()
    normalized = keypoints - wrist
    scale = np.linalg.norm(normalized[9]) + 1e-6
    return normalized / scale


def extract_hand_angles(keypoints):
    """Extract 10 joint angles from 21 keypoints (numpy, no torch needed)."""
    joint_list = [[3, 2, 1], [7, 6, 5], [11, 10, 9], [15, 14, 13], [19, 18, 17]]
    angles = []
    for joint in joint_list:
        a = keypoints[joint[0]]
        b = keypoints[joint[1]]
        c = keypoints[joint[2]]
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        angles.append(angle / 180.0)
    # Also add 5 finger extension ratios (tip-to-wrist vs mcp-to-wrist)
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    for tip_idx, mcp_idx in zip(tips, mcps):
        tip_dist = np.linalg.norm(keypoints[tip_idx] - keypoints[0])
        mcp_dist = np.linalg.norm(keypoints[mcp_idx] - keypoints[0]) + 1e-6
        angles.append(min(tip_dist / mcp_dist, 2.0) / 2.0)
    return np.array(angles, dtype=np.float32)  # [10]


def extract_hand_angles_torch(keypoints):
    """Extract 10 joint angles from keypoints (PyTorch, differentiable).

    keypoints: [B, 21, 3]
    Returns: [B, 10]
    """
    joint_list = [[3, 2, 1], [7, 6, 5], [11, 10, 9], [15, 14, 13], [19, 18, 17]]
    angles = []
    for a_idx, b_idx, c_idx in joint_list:
        a = keypoints[:, a_idx, :2]
        b = keypoints[:, b_idx, :2]
        c = keypoints[:, c_idx, :2]
        ba = a - b
        bc = c - b
        angle = torch.atan2(bc[:, 1], bc[:, 0]) - torch.atan2(ba[:, 1], ba[:, 0])
        angle = torch.abs(torch.rad2deg(angle))
        angle = torch.where(angle > 180.0, 360.0 - angle, angle)
        angles.append(angle / 180.0)
    # Finger extension ratios
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    for tip_idx, mcp_idx in zip(tips, mcps):
        tip_dist = torch.norm(keypoints[:, tip_idx] - keypoints[:, 0], dim=1)
        mcp_dist = torch.norm(keypoints[:, mcp_idx] - keypoints[:, 0], dim=1) + 1e-6
        ratio = torch.clamp(tip_dist / mcp_dist, max=2.0) / 2.0
        angles.append(ratio)
    return torch.stack(angles, dim=1)  # [B, 10]


# ============================================================
# PyTorch GCN model
# ============================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer with edge-type awareness."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        """x: [B, N, C], edge_index: (tensor[E], tensor[E]) on same device as x"""
        B, N, _ = x.shape
        edge_from, edge_to = edge_index
        out = torch.zeros(B, N, self.weight.shape[1], device=x.device, dtype=x.dtype)
        for i in range(N):
            mask = edge_to == i
            neighbors = edge_from[mask]
            if neighbors.numel() > 0:
                neighbor_feats = x[:, neighbors, :].mean(dim=1)
                out[:, i, :] = neighbor_feats @ self.weight + self.bias
            else:
                out[:, i, :] = x[:, i, :] @ self.weight + self.bias
        return out


class SpatialGCN(nn.Module):
    """Graph Convolutional Network for 21-point hand topology encoding.

    Input:  [B, 21, 3]  normalized keypoints
    Output: [B, 256]    spatial topology features
    Params: ~0.15M

    Uses hand-optimized topology by default (HA-GCN [9] + Cross Attentive [24]):
    fingertip mesh + intra-finger skip edges + inter-finger cross-links.
    """

    def __init__(self, in_channels=3, hidden_dim=128, out_features=256,
                 optimized_topology=True, angle_dim=64):
        super().__init__()
        ei = HandSkeletonGraph.EDGE_INDEX_OPTIMIZED if optimized_topology else HandSkeletonGraph.EDGE_INDEX
        edge_from = torch.tensor(ei[0], dtype=torch.long)
        edge_to = torch.tensor(ei[1], dtype=torch.long)
        self.register_buffer('_edge_from', edge_from)
        self.register_buffer('_edge_to', edge_to)

        self.gcn1 = GraphConvLayer(in_channels, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvLayer(hidden_dim, hidden_dim)

        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.ReLU(),
            nn.LayerNorm(out_features)
        )

        # Angle feature encoder (10 features: 5 joint angles + 5 extension ratios)
        self.angle_encoder = nn.Sequential(
            nn.Linear(10, angle_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(angle_dim, angle_dim)
        )

        # Residual connection for angle features
        self.final_fusion = nn.Sequential(
            nn.Linear(out_features + angle_dim, out_features),
            nn.ReLU(),
            nn.LayerNorm(out_features)
        )
        self.dropout = nn.Dropout(0.1)

    @property
    def edge_index(self):
        return (self._edge_from, self._edge_to)

    def forward(self, keypoints):
        """keypoints: [B, 21, 3] normalized coordinates."""
        B = keypoints.shape[0]
        x = keypoints

        # GCN layers with residuals
        x1 = F.relu(self.gcn1(x, self.edge_index))
        x1 = self.dropout(x1)
        x2 = F.relu(self.gcn2(x1, self.edge_index)) + x1
        x2 = self.dropout(x2)
        x3 = F.relu(self.gcn3(x2, self.edge_index)) + x2

        # Global pooling
        pooled = x3.mean(dim=1)  # [B, hidden_dim]
        spatial_feat = self.pool_proj(pooled)  # [B, 256]

        # Angle features (differentiable — pure PyTorch path)
        angle_tensor = extract_hand_angles_torch(keypoints)  # [B, 10]
        angle_feat = self.angle_encoder(angle_tensor)        # [B, 64]

        # Fusion
        fused = torch.cat([spatial_feat, angle_feat], dim=1)
        return self.final_fusion(fused)  # [B, 256]
