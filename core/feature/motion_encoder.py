# coding:utf-8
"""Motion encoder: optical flow histogram + keypoint velocity features."""
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def compute_flow_histogram(flow_field, bins=64):
    """Extract 128-dim histogram from optical flow field (numpy, no torch needed).
    flow_field: [H, W, 2]
    Returns: [128] magnitude + direction histogram (64 mag bins + 64 ang bins)
    """
    mag, ang = cv2.cartToPolar(flow_field[..., 0], flow_field[..., 1])
    mag = mag.flatten()
    ang = ang.flatten() * 180 / np.pi

    mag_hist, _ = np.histogram(mag, bins=bins, range=(0, 10))
    ang_hist, _ = np.histogram(ang, bins=bins, range=(0, 360))

    mag_hist = mag_hist.astype(np.float32) / (mag_hist.sum() + 1e-6)
    ang_hist = ang_hist.astype(np.float32) / (ang_hist.sum() + 1e-6)

    return np.concatenate([mag_hist, ang_hist])


try:
    import cv2
except ImportError:
    cv2 = None


def extract_motion_features(prev_keypoints, curr_keypoints):
    """Extract keypoint velocity and acceleration features.
    prev_keypoints: [21, 3] or None
    curr_keypoints: [21, 3]
    Returns: [63] velocity vector (21*3 flattened)
    """
    curr = np.array([[lm.x, lm.y, lm.z] if hasattr(lm, 'x') else lm
                      for lm in curr_keypoints], dtype=np.float32)
    if prev_keypoints is None:
        return np.zeros(63, dtype=np.float32)

    prev = np.array([[lm.x, lm.y, lm.z] if hasattr(lm, 'x') else lm
                      for lm in prev_keypoints], dtype=np.float32)
    velocity = (curr - prev).flatten()
    return velocity.astype(np.float32)


class MotionEncoder(nn.Module):
    """Encodes optical flow histogram + keypoint motion features.

    Input:  flow_hist [B, 128], keypoint_diff [B, 63]
    Output: [B, 128] motion feature
    Params: ~0.04M
    """

    def __init__(self, flow_dim=128, kp_dim=63, out_features=128):
        super().__init__()
        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_dim, 96),
            nn.ReLU(),
            nn.LayerNorm(96)
        )
        self.kp_encoder = nn.Sequential(
            nn.Linear(kp_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        # Learnable fusion weights (2-way)
        self.fusion_weight_flow = nn.Parameter(torch.tensor(0.5))
        self.fusion_weight_kp = nn.Parameter(torch.tensor(0.5))
        self.fusion = nn.Sequential(
            nn.Linear(96 + 64, out_features),
            nn.ReLU(),
            nn.LayerNorm(out_features)
        )

    def forward(self, flow_hist, keypoint_diff, kinematic_vec=None):
        """flow_hist: [B, 128], keypoint_diff: [B, 63]
        kinematic_vec: ignored (backward compat for checkpoint loading)"""
        f_feat = self.flow_encoder(flow_hist)    # [B, 96]
        k_feat = self.kp_encoder(keypoint_diff)   # [B, 64]

        # 2-way adaptive fusion with learned weights
        w = torch.softmax(torch.stack([
            self.fusion_weight_flow,
            self.fusion_weight_kp,
        ]), dim=0)

        fused = torch.cat([f_feat * w[0], k_feat * w[1]], dim=1)
        return self.fusion(fused)  # [B, 128]
