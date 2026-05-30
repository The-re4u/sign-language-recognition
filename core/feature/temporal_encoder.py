# coding:utf-8
"""TemporalEncoder: frame-to-frame trajectory encoder for MediaPipe landmarks.

Extracts per-frame motion statistics (velocity, acceleration, curvature,
global motion magnitude) from keypoint sequences and projects to 256-dim
embedding. Replaces the old Farneback optical-flow-based MotionEncoder.

Reference: docs/多模态方案升级说明.md Section 5.3
"""
import torch
import torch.nn as nn


def extract_trajectory_features(kp_seq, fps=25.0):
    """Extract per-frame trajectory features from keypoint sequence.

    Args:
        kp_seq: [T, 21, 3] or [B, T, 21, 3] Image Landmarks sequence
        fps: frames per second (for optional time-normalization)

    Returns:
        features: [T, 106] or [B, T, 106] per-frame motion descriptors.
                  First frame is zeros. T=1 returns all zeros.
    """
    single = kp_seq.dim() == 3
    if single:
        kp_seq = kp_seq.unsqueeze(0)  # [1, T, 21, 3]

    B, T, _, _ = kp_seq.shape
    xy = kp_seq[:, :, :, :2]  # [B, T, 21, 2]

    feats = []
    for b in range(B):
        frame_feats = []
        # Frame 0: all zeros
        frame_feats.append(torch.zeros(106, device=kp_seq.device, dtype=kp_seq.dtype))

        for t in range(1, T):
            # Velocity: 21 keypoints × 2D displacement
            vel = xy[b, t] - xy[b, t - 1]                       # [21, 2]

            # Acceleration: velocity change
            if t >= 2:
                vel_prev = xy[b, t - 1] - xy[b, t - 2]
                acc = vel - vel_prev                              # [21, 2]
            else:
                acc = torch.zeros(21, 2, device=kp_seq.device, dtype=kp_seq.dtype)

            # Curvature: direction change per keypoint
            vel_norm = vel.norm(dim=1) + 1e-8                     # [21]
            vel_dir = vel / vel_norm.unsqueeze(1)                 # [21, 2]
            if t >= 2:
                prev_norm = vel_prev.norm(dim=1) + 1e-8
                prev_dir = vel_prev / prev_norm.unsqueeze(1)
                curv = 1.0 - (vel_dir * prev_dir).sum(dim=1)     # [21]
            else:
                curv = torch.zeros(21, device=kp_seq.device, dtype=kp_seq.dtype)

            # Global motion magnitude
            motion_mag = vel_norm.mean().unsqueeze(0)             # [1]

            # Concatenate: 42 + 42 + 21 + 1 = 106
            f = torch.cat([
                vel.flatten(),     # 42
                acc.flatten(),     # 42
                curv,              # 21
                motion_mag,        # 1
            ])
            frame_feats.append(f)

        feats.append(torch.stack(frame_feats))  # [T, 106]

    out = torch.stack(feats)  # [B, T, 106]

    if single:
        out = out.squeeze(0)
    return out


class TemporalEncoder(nn.Module):
    """Frame-to-frame trajectory encoder: keypoint seq → motion embedding.

    Extracts 106-dim per-frame motion descriptors (velocity, acceleration,
    curvature, global motion) and projects to 256-dim via MLP.
    Includes per-keypoint temporal attention (SKIM [35]) for adaptive
    weighting of discriminative keypoints.

    Input:  [B, T, 21, 3]
    Output: [B, T, 256]
    Params: ~0.2M
    """

    def __init__(self, feature_dim=106, hidden_dim=128, out_features=256,
                 dropout=0.1):
        super().__init__()
        # Per-keypoint attention: keypoint features → importance weight
        self.point_attn = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
        )
        self.out_features = out_features

    def forward(self, kp):
        """Extract trajectory features with per-keypoint attention, then project.

        Args:
            kp: [B, T, 21, 3]

        Returns:
            embedding: [B, T, 256]
        """
        B, T = kp.shape[:2]

        traj = extract_trajectory_features(kp)  # [B, T, 106]

        # Per-keypoint attention from raw keypoints (SKIM [35])
        kp_flat = kp.reshape(B * T, 21, 3)
        attn_logits = self.point_attn(kp_flat)            # [B*T, 21, 1]
        attn = torch.softmax(attn_logits, dim=1)           # [B*T, 21, 1]
        attn = attn.reshape(B, T, 21, 1)

        # Apply to velocity (dims 0-41) and acceleration (dims 42-83)
        vel = traj[:, :, :42].reshape(B, T, 21, 2)
        vel = (vel * attn).reshape(B, T, 42)

        acc = traj[:, :, 42:84].reshape(B, T, 21, 2)
        acc = (acc * attn).reshape(B, T, 42)

        traj_weighted = torch.cat([
            vel,                       # 42 (weighted)
            acc,                       # 42 (weighted)
            traj[:, :, 84:105],        # 21 (curvature, unchanged)
            traj[:, :, 105:106],       # 1  (motion_magnitude, unchanged)
        ], dim=-1)

        traj_flat = traj_weighted.reshape(B * T, 106)
        emb = self.mlp(traj_flat)                       # [B*T, 256]
        return emb.reshape(B, T, self.out_features)
