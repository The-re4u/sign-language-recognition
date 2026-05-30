# coding:utf-8
"""HandShapeContext: geometric descriptor encoder for MediaPipe landmarks.

Extracts 29-dim hand shape descriptors from 21 Image Landmarks and projects
to 256-dim embedding via MLP. Designed as a drop-in replacement for the
CNN-based visual encoder — captures finger angles, spread, bone ratios,
fingertip distribution, and thumb abduction as interpretable geometric features.

Reference: docs/多模态方案升级说明.md Section 5.2
"""
import torch
import torch.nn as nn
import numpy as np


# ── MediaPipe landmark indices ──────────────────────────────────────────
WRIST = 0
THUMB = [1, 2, 3, 4]      # CMC, MCP, IP, TIP
INDEX = [5, 6, 7, 8]      # MCP, PIP, DIP, TIP
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]

FINGERS = [THUMB, INDEX, MIDDLE, RING, PINKY]
FINGER_MCP = [2, 5, 9, 13, 17]    # MCP for each finger
FINGER_TIP = [4, 8, 12, 16, 20]   # TIP for each finger


def extract_hand_shape_descriptors(kp):
    """Extract 29-dim geometric descriptors from Image Landmarks.

    Args:
        kp: [..., 21, 3] normalized Image Landmarks (x,y in [0,1], z relative)

    Returns:
        descriptors: [..., 29] geometric feature vector
    """
    orig_shape = kp.shape
    kp = kp.reshape(-1, 21, 3)  # [N, 21, 3]
    N = kp.shape[0]
    device = kp.device

    # Use 2D coordinates (xy) for angles; z for depth cues
    xy = kp[:, :, :2]  # [N, 21, 2]
    z = kp[:, :, 2]    # [N, 21]

    features = []

    # ── 1. Finger straightness angles (5-dim) ──
    # Angle at the middle joint: MCP→PIP vs PIP→DIP (MCP→IP vs IP→TIP for thumb)
    joints = [
        (2, 3, 4),    # Thumb: MCP→IP, IP→TIP
        (5, 6, 7),    # Index: MCP→PIP, PIP→DIP
        (9, 10, 11),  # Middle
        (13, 14, 15), # Ring
        (17, 18, 19), # Pinky
    ]
    for a, b, c in joints:
        v1 = xy[:, b] - xy[:, a]  # [N, 2] proximal segment
        v2 = xy[:, c] - xy[:, b]  # [N, 2] middle segment
        cos_a = (v1 * v2).sum(dim=1) / (v1.norm(dim=1) * v2.norm(dim=1) + 1e-8)
        angle_rad = torch.acos(cos_a.clamp(-1, 1))
        features.append(angle_rad.unsqueeze(1))  # [N, 1]

    # ── 2. Inter-finger spread angles (4-dim) ──
    # Angle between adjacent MCP vectors (from wrist)
    for i in range(len(FINGER_MCP) - 1):
        v1 = xy[:, FINGER_MCP[i]] - xy[:, WRIST]    # [N, 2]
        v2 = xy[:, FINGER_MCP[i + 1]] - xy[:, WRIST]
        cos_a = (v1 * v2).sum(dim=1) / (v1.norm(dim=1) * v2.norm(dim=1) + 1e-8)
        angle_rad = torch.acos(cos_a.clamp(-1, 1))
        features.append(angle_rad.unsqueeze(1))

    # ── 3. Bone length ratios (5-dim) ──
    # proximal_length / total_finger_length per finger
    segments = [
        (1, 2, 3),     # Thumb: CMC→MCP, MCP→IP
        (5, 6, 7),     # Index: MCP→PIP, PIP→DIP
        (9, 10, 11),   # Middle
        (13, 14, 15),  # Ring
        (17, 18, 19),  # Pinky
    ]
    hand_span = (xy.max(dim=1).values - xy.min(dim=1).values).norm(dim=1) + 1e-8  # [N]
    for p1, p2, p3 in segments:
        l1 = (xy[:, p2] - xy[:, p1]).norm(dim=1)  # proximal length
        l2 = (xy[:, p3] - xy[:, p2]).norm(dim=1)  # middle length
        ratio = l1 / (l1 + l2 + 1e-8)
        features.append(ratio.unsqueeze(1))

    # ── 4. Fingertip distribution (10-dim) ──
    # Each fingertip relative to wrist: direction (dx,dy normalized) + distance
    for tip_idx in FINGER_TIP:
        vec = xy[:, tip_idx] - xy[:, WRIST]          # [N, 2]
        dist = vec.norm(dim=1) / hand_span           # [N]
        vec_norm = vec / (vec.norm(dim=1, keepdim=True) + 1e-8)
        features.append(vec_norm[:, 0:1])             # dx direction
        features.append(dist.unsqueeze(1))            # distance (normalized)

    # ── 5. Thumb abduction (2-dim) ──
    # Angle between thumb MCP vector and middle MCP vector (both from wrist)
    v_thumb = xy[:, FINGER_MCP[0]] - xy[:, WRIST]     # thumb
    v_mid = xy[:, FINGER_MCP[2]] - xy[:, WRIST]       # middle
    cos_ab = (v_thumb * v_mid).sum(dim=1) / (v_thumb.norm(dim=1) * v_mid.norm(dim=1) + 1e-8)
    ab_angle = torch.acos(cos_ab.clamp(-1, 1))
    ab_mag = v_thumb.norm(dim=1) / hand_span
    features.append(ab_angle.unsqueeze(1))
    features.append(ab_mag.unsqueeze(1))

    # ── 6. MCP convex hull area (1-dim) ──
    # Area of polygon formed by 5 MCP points, normalized by hand span^2
    mcp_pts = xy[:, FINGER_MCP]  # [N, 5, 2]
    # Shoelace formula
    x, y = mcp_pts[:, :, 0], mcp_pts[:, :, 1]
    area = 0.5 * torch.abs(
        (x[:, 0] * y[:, 1] + x[:, 1] * y[:, 2] + x[:, 2] * y[:, 3] +
         x[:, 3] * y[:, 4] + x[:, 4] * y[:, 0]) -
        (y[:, 0] * x[:, 1] + y[:, 1] * x[:, 2] + y[:, 2] * x[:, 3] +
         y[:, 3] * x[:, 4] + y[:, 4] * x[:, 0])
    )
    area_norm = area / (hand_span * hand_span + 1e-8)
    features.append(area_norm.unsqueeze(1))

    # ── 7. Bent-finger z deviation (2-dim) ──
    # PIP.z - DIP.z for index and middle — captures side-hook depth cue
    for mcp, pip, dip in [(5, 6, 7), (9, 10, 11)]:
        z_dev = z[:, pip] - z[:, dip]
        features.append(z_dev.unsqueeze(1))

    desc = torch.cat(features, dim=1)  # [N, 29]

    if len(orig_shape) > 2:
        new_shape = orig_shape[:-2] + (29,)
        desc = desc.reshape(new_shape)
    return desc


class MiniPointNet(nn.Module):
    """Lightweight PointNet on 21×3 keypoints — learned spatial features.

    Fraunhofer (2025) [36]: 3D geometric features + PointNet fusion on
    MediaPipe hand skeleton. ~50K params.
    """

    def __init__(self, in_channels=3, hidden_dim=64, out_features=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, out_features, 1),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, kp):
        """kp: [N, 21, 3] → [N, 128]"""
        kp_t = kp.transpose(-1, -2)        # [N, 3, 21]
        feat = self.conv(kp_t)             # [N, 128, 21]
        feat = self.pool(feat).squeeze(-1) # [N, 128]
        return feat


class HandShapeContext(nn.Module):
    """Geometric dual-stream encoder: hand-crafted descriptors + MiniPointNet.

    Stream A: 29-dim interpretable descriptors (finger angles, spread, bone
              ratios, fingertip distribution, etc.) → MLP → 128-dim
    Stream B: MiniPointNet on raw 21×3 keypoints → 128-dim (learned)
    Fusion:   Concat → Linear → 256-dim

    Design follows Fraunhofer (2025) [36]: geometric features MLP + PointNet
    fusion on MediaPipe hand skeleton.

    Input:  [B, 21, 3] or [B, T, 21, 3]
    Output: [B, 256] or [B, T, 256]
    Params: ~0.2M
    """

    def __init__(self, descriptor_dim=29, hidden_dim=128, out_features=256,
                 pointnet_dim=128, dropout=0.1):
        super().__init__()
        # Stream A: hand-crafted descriptors
        self.desc_mlp = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stream B: MiniPointNet (learned from raw keypoints)
        self.pointnet = MiniPointNet(in_channels=3, out_features=pointnet_dim)

        # Fusion: concat → 256
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + pointnet_dim, out_features),
            nn.LayerNorm(out_features),
        )
        self.out_features = out_features

    def forward(self, kp):
        """Extract dual-stream geometric features.

        Args:
            kp: [B, 21, 3] or [B, T, 21, 3]

        Returns:
            embedding: [B, 256] or [B, T, 256]
        """
        has_time = kp.dim() == 4
        if has_time:
            B, T = kp.shape[:2]
            kp_flat = kp.reshape(B * T, 21, 3)
        else:
            kp_flat = kp

        # Stream A: hand-crafted descriptors
        desc = extract_hand_shape_descriptors(kp_flat)  # [N, 29]
        desc_out = self.desc_mlp(desc)                   # [N, 128]

        # Stream B: MiniPointNet on raw keypoints
        pn_out = self.pointnet(kp_flat)                  # [N, 128]

        # Fusion
        fused = self.fusion(torch.cat([desc_out, pn_out], dim=-1))  # [N, 256]

        if has_time:
            fused = fused.reshape(B, T, self.out_features)
        return fused
