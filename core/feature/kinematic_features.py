# coding:utf-8
"""
Explicit kinematic feature extraction — validated metrics from:
  Wagh et al. (2025) "Using MediaPipe to track upper-limb reaching movements
  after stroke." J NeuroEngineering Rehabil, 22, 268.

Computes palm speed, fingertip velocities, inter-finger coordination (BVE-like),
and trajectory stability — all physically meaningful and directly interpretable
features that supplement the learned deep representations.
"""
import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17
TIP_INDICES = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
MCP_INDICES = [2, 5, 9, 13, 17]     # thumb, index, middle, ring, pinky
PIP_INDICES = [3, 6, 10, 14, 18]    # proximal interphalangeal (highest MediaPipe error)

# Fingers in order: thumb, index, middle, ring, pinky
FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']


def palm_center(keypoints):
    """Palm center = centroid of wrist + index_mcp + pinky_mcp (paper eq.).
    keypoints: [21, 2] or [21, 3]"""
    return (keypoints[WRIST] + keypoints[INDEX_MCP] + keypoints[PINKY_MCP]) / 3.0


def compute_palm_speed(prev_kp, curr_kp, fps=30):
    """Mean palm speed — paper's primary kinematic outcome.
    Returns scalar (normalized units / second)."""
    prev_palm = palm_center(prev_kp[:, :2])
    curr_palm = palm_center(curr_kp[:, :2])
    return np.linalg.norm(curr_palm - prev_palm) * fps


def compute_tip_velocities(prev_kp, curr_kp, fps=30):
    """Per-fingertip velocity vector magnitudes.
    Returns [5] array — one scalar per finger."""
    velocities = []
    for tip in TIP_INDICES:
        vel = np.linalg.norm(curr_kp[tip, :2] - prev_kp[tip, :2]) * fps
        velocities.append(vel)
    return np.array(velocities, dtype=np.float32)


def compute_finger_coordination(keypoints, metric='pairwise_spread'):
    """Inter-finger coordination: pairwise fingertip distance change rate.
    Simplified from the paper's BVE — measures how fingers move relative to
    each other (high during opening/closing, low during static holds).

    keypoints: [21, 2]
    Returns [10] array for C(5,2) fingertip pairs."""
    n_tips = len(TIP_INDICES)
    n_pairs = n_tips * (n_tips - 1) // 2
    coordination = np.zeros(n_pairs, dtype=np.float32)
    idx = 0
    for i in range(n_tips):
        for j in range(i + 1, n_tips):
            dist = np.linalg.norm(keypoints[TIP_INDICES[i], :2] -
                                  keypoints[TIP_INDICES[j], :2])
            coordination[idx] = dist
            idx += 1
    return coordination


def compute_coordination_change(prev_kp, curr_kp, fps=30):
    """Rate of change in inter-finger coordination (BVE-like).
    Returns [10] array — spread rate for each fingertip pair."""
    prev_coord = compute_finger_coordination(prev_kp[:, :2])
    curr_coord = compute_finger_coordination(curr_kp[:, :2])
    return np.abs(curr_coord - prev_coord) * fps


def compute_trajectory_stability(kp_history, window=16):
    """BVE-style trajectory stability over a window of recent keypoints.
    Lower = more stable (less jitter).

    kp_history: list of [21, 2] arrays (newest last)
    Returns: scalar stability score [0, 1], 1 = perfectly stable."""
    if len(kp_history) < 2:
        return 1.0

    recent = kp_history[-window:] if len(kp_history) >= window else kp_history
    stacked = np.stack(recent, axis=0)  # [W, 21, 2]
    mean_traj = stacked.mean(axis=0)     # [21, 2]
    deviations = np.linalg.norm(stacked - mean_traj, axis=2)  # [W, 21]
    bve = deviations.mean()  # mean deviation across all landmarks and frames

    # Map BVE to [0,1] stability — gentler scaling for active gestures
    stability = 1.0 / (1.0 + bve * 5.0)
    return float(np.clip(stability, 0.0, 1.0))


def compute_joint_reliability_weights():
    """Per-landmark reliability weights based on literature.
    PIP joints (6,10,14,18) and thumb CMC (1) have highest reported error.
    Returns [21] array of weights in [0.6, 1.0]."""
    weights = np.ones(21, dtype=np.float32)

    # PIP joints: highest MediaPipe error (Sprague 2025, Maggioni 2025)
    for pip in PIP_INDICES:
        weights[pip] = 0.65

    # Thumb CMC and MCP: second-highest error
    weights[1] = 0.75
    weights[2] = 0.80

    # Fingertips: moderate error
    for tip in TIP_INDICES:
        weights[tip] = 0.85

    # Wrist and MCP bases: most reliable
    # weights[0,5,9,13,17] = 1.0 (default)

    return weights


def compute_all_kinematic_features(prev_kp, curr_kp, fps=30, kp_history=None):
    """Compute all validated kinematic features for a frame pair.

    Args:
        prev_kp: [21, 3] or [21, 2] previous keypoints
        curr_kp: [21, 3] or [21, 2] current keypoints
        fps: frames per second
        kp_history: optional list of recent [21,2] arrays for stability

    Returns:
        dict with keys: palm_speed, tip_velocities, coordination_change,
                        trajectory_stability, joint_weights
    """
    prev_2d = prev_kp[:, :2].astype(np.float32)
    curr_2d = curr_kp[:, :2].astype(np.float32)

    features = {
        'palm_speed': np.array([compute_palm_speed(prev_2d, curr_2d, fps)],
                               dtype=np.float32),
        'tip_velocities': compute_tip_velocities(prev_2d, curr_2d, fps),
        'coordination_change': compute_coordination_change(prev_2d, curr_2d, fps),
        'trajectory_stability': np.array(
            [compute_trajectory_stability(kp_history) if kp_history else 1.0],
            dtype=np.float32),
        'joint_weights': compute_joint_reliability_weights(),
    }

    # Flattened concatenation for model input: 1 + 5 + 10 + 1 = 17 dims
    features['vector'] = np.concatenate([
        features['palm_speed'],
        features['tip_velocities'],
        features['coordination_change'],
        features['trajectory_stability'],
    ]).astype(np.float32)

    return features


# ============================================================
# PyTorch kinematic encoder module
# ============================================================
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class KinematicFeatureEncoder(nn.Module):
    """Encodes explicit kinematic features (17-dim) into a 64-dim embedding.

    Input:  [B, 17] kinematic feature vector
    Output: [B, 64]  kinematic embedding
    Params: ~0.005M
    """

    def __init__(self, input_dim=17, out_features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Linear(48, out_features),
            nn.ReLU(),
            nn.LayerNorm(out_features),
        )

    def forward(self, x):
        return self.encoder(x)
