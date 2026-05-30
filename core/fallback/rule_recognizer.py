# coding:utf-8
"""
Rule-based gesture recognition — CSL (Chinese Sign Language) digits 0-9.
Serves as fallback when deep learning model is unavailable.

v5.0: Redesigned gesture definitions with multi-angle support.
  0 — fist (all curled), front only; side fist falls through to One
  1 — single finger (thumb or index), any angle
  2 — index + middle (wide/medium/together)
  3 — any 3 consecutive fingers (T+I+M / I+M+R / M+R+P)
  4 — only thumb bent, other 4 extended
  5 — all 5 extended
  6 — thumb + pinky
  7 — thumb + index + pinky, middle + ring curled (2D only)
  8 — thumb + index L-shape (thumb truly abducted, thumb-index angle 20-80)
  9 — hooked index, front & side (multi-angle: PIP bend or z-gradient)

v4.0: CSL 0-9 digit gestures only, control gestures handled by SentenceRecorder.
v2.1: PIP joint error compensation (Wagh et al. 2025).
"""
import numpy as np

# Landmark indices
MCP_JOINTS = [2, 5, 9, 13, 17]   # Metacarpophalangeal (most reliable)
PIP_JOINTS = [3, 6, 10, 14, 18]  # Proximal interphalangeal (highest error)
TIP_JOINTS = [4, 8, 12, 16, 20]  # Fingertips (moderate error)

# Gestures that depend heavily on PIP joint accuracy
PIP_SENSITIVE_GESTURES = {'Two', 'Three', 'Four', 'Six', 'Nine'}


def cal_angle(a, b, c):
    """Calculate 2D angle at point b formed by points a-b-c (x, y only)."""
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def cal_angle_3d(a, b, c):
    """Calculate 3D angle at point b formed by points a-b-c (x, y, z)."""
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]], dtype=np.float64)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def _landmark_3d(lm, idx):
    """Return (x, y, z) tuple for landmark at index."""
    return (lm.landmark[idx].x, lm.landmark[idx].y, lm.landmark[idx].z)


def each_finger_up(hand_landmarks):
    """Determine which fingers are extended (angle > 165 degrees)."""
    joint_list = [[3, 2, 1], [7, 6, 5], [11, 10, 9], [15, 14, 13], [19, 18, 17]]
    is_finger_up = []
    for joint in joint_list:
        a = np.array([hand_landmarks.landmark[joint[0]].x, hand_landmarks.landmark[joint[0]].y])
        b = np.array([hand_landmarks.landmark[joint[1]].x, hand_landmarks.landmark[joint[1]].y])
        c = np.array([hand_landmarks.landmark[joint[2]].x, hand_landmarks.landmark[joint[2]].y])
        angle = cal_angle(a, b, c)
        is_finger_up.append(angle > 165)
    return is_finger_up


def rec_gesture(is_finger_up, hand_landmarks, pinch_threshold=0.05):
    """CSL 0-9 digit gesture recognition (v5.1 — semantic output names).

    Returns gesture names matching SentenceRecorder expectations:
      Closed_Fist(0) — fist (all fingers curled)
      One(1)         — single finger extended (thumb/index/pinky)
      Two(2)         — index + middle (3 variants: wide/medium/together)
      Three(3)       — any 3 consecutive fingers (T+I+M / I+M+R / M+R+P)
      Four(4)        — only thumb bent, other 4 extended
      Open_Palm(5)   — all 5 fingers extended
      Six(6)         — thumb + pinky
      Seven(7)       — thumb + index + pinky, middle + ring curled (2D only)
      Eight(8)       — thumb + index L-shape (thumb truly abducted)
      Nine(9)        — hooked index, front & side (multi-angle detection)

    Args:
        is_finger_up: [thumb, index, middle, ring, pinky] bool list
        hand_landmarks: MediaPipe hand landmarks
        pinch_threshold: unused (kept for API compatibility)
    """
    T, I, M, R, P = is_finger_up
    n = sum(is_finger_up)

    # ============================================================
    # CSL 9: hooked index (checked first — multi-angle detection)
    # ============================================================
    mcp_idx = cal_angle_3d(
        _landmark_3d(hand_landmarks, 6),   # index PIP
        _landmark_3d(hand_landmarks, 5),   # index MCP
        _landmark_3d(hand_landmarks, 0),   # wrist
    )
    pip_idx = cal_angle_3d(
        _landmark_3d(hand_landmarks, 7),   # index DIP
        _landmark_3d(hand_landmarks, 6),   # index PIP
        _landmark_3d(hand_landmarks, 5),   # index MCP
    )
    dip_idx = cal_angle_3d(
        _landmark_3d(hand_landmarks, 8),   # index TIP
        _landmark_3d(hand_landmarks, 7),   # index DIP
        _landmark_3d(hand_landmarks, 6),   # index PIP
    )
    mid_mcp = cal_angle_3d(
        _landmark_3d(hand_landmarks, 10), _landmark_3d(hand_landmarks, 9), _landmark_3d(hand_landmarks, 0))
    ring_mcp = cal_angle_3d(
        _landmark_3d(hand_landmarks, 14), _landmark_3d(hand_landmarks, 13), _landmark_3d(hand_landmarks, 0))
    pinky_mcp = cal_angle_3d(
        _landmark_3d(hand_landmarks, 18), _landmark_3d(hand_landmarks, 17), _landmark_3d(hand_landmarks, 0))

    others_up = sum(is_finger_up[i] for i in [0, 2, 3, 4])  # thumb/mid/ring/pinky up in 2D
    others_mcp_ext = sum(1 for a in [mid_mcp, ring_mcp, pinky_mcp] if a > 140)

    # Path 1 (side view): MCP extended + PIP moderately bent (hook, not full curl)
    # Lower bound pip > 50 excludes fully curled fists (PIP ~15 deg) from being
    # misdetected as Nine.
    hook_side = mcp_idx > 140 and 50 < pip_idx < 130

    # Path 2 (front view): MCP extended, PIP looks straight from front,
    # but z-gradient and short tip-to-MCP distance reveal the forward hook.
    # pip_idx >= 130 ensures the finger is NOT fully curled (excludes fists,
    # where pip ~15 deg also has short tip distance and positive z-gradient).
    hook_front = False
    if not hook_side and mcp_idx > 150 and pip_idx >= 130 and others_up <= 1:
        tip = np.array(_landmark_3d(hand_landmarks, 8))
        mcp = np.array(_landmark_3d(hand_landmarks, 5))
        tip_mcp_3d = np.linalg.norm(tip - mcp)
        z_grad = tip[2] - mcp[2]
        hook_front = (dip_idx < 135 or (tip_mcp_3d < 0.22 and z_grad > 0.008))

    if hook_side and others_up <= 1 and others_mcp_ext <= 1:
        return 'Nine'
    if hook_front:
        return 'Nine'

    # ============================================================
    # CSL 8 (early check): thumb + index L-shape from the side.
    # From the side, the index PIP 2D may fall just below 165 deg,
    # making n=1 (thumb only). Check thumb 3D extension + thumb-index
    # abduction angle (L-shape = 20-80 deg between thumb and index vectors).
    # ============================================================
    t_mcp_3d_early = cal_angle_3d(
        _landmark_3d(hand_landmarks, 0),   # wrist
        _landmark_3d(hand_landmarks, 2),   # thumb MCP
        _landmark_3d(hand_landmarks, 3),   # thumb IP
    )
    # Index 2D PIP angle for "almost up" detection
    idx_pip_2d = cal_angle(
        (hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y),   # index MCP
        (hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y),   # index PIP
        (hand_landmarks.landmark[7].x, hand_landmarks.landmark[7].y),   # index DIP
    )
    if t_mcp_3d_early > 155 and not M and not P and ((T and I) or (T and idx_pip_2d > 140)):
        t_tip = np.array(_landmark_3d(hand_landmarks, 4))
        t_mcp_v = np.array(_landmark_3d(hand_landmarks, 2))
        i_tip = np.array(_landmark_3d(hand_landmarks, 8))
        i_mcp_v = np.array(_landmark_3d(hand_landmarks, 5))
        v_t = t_tip - t_mcp_v
        v_i = i_tip - i_mcp_v
        cos_abd = np.dot(v_t, v_i) / (np.linalg.norm(v_t) * np.linalg.norm(v_i) + 1e-8)
        thumb_abd = np.degrees(np.arccos(np.clip(cos_abd, -1.0, 1.0)))
        if 20 < thumb_abd < 80:
            return 'Eight'

    # ============================================================
    # CSL 0-8: count-based classification
    # ============================================================

    # --- Side-view One: thumb extended but PIP 2D < 165 (n=0) ---
    # From the side, the thumb PIP 2D angle may fall below the 165 deg
    # threshold. Use 3D thumb extension as fallback.
    if n == 0:
        # Check if thumb is extended in 3D (wrist -> MCP -> IP)
        thumb_3d_ext = cal_angle_3d(
            _landmark_3d(hand_landmarks, 0),   # wrist
            _landmark_3d(hand_landmarks, 2),   # thumb MCP
            _landmark_3d(hand_landmarks, 3),   # thumb IP
        )
        if thumb_3d_ext > 155:
            return 'One'
        return 'Closed_Fist'

    if n == 1:
        return 'One'

    if n == 2:
        # Six: thumb + pinky
        if T and P:
            return 'Six'

        # Eight: thumb + index L-shape (thumb truly abducted from index).
        # Distinguishes from 1_index_side where thumb falsely registers as
        # "up" in 2D but is alongside the index (thumb-index angle < 25 deg).
        if T and I:
            t_mcp_3d = cal_angle_3d(
                _landmark_3d(hand_landmarks, 0),   # wrist
                _landmark_3d(hand_landmarks, 2),   # thumb MCP
                _landmark_3d(hand_landmarks, 3),   # thumb IP
            )
            t_tip = np.array(_landmark_3d(hand_landmarks, 4))
            t_mcp = np.array(_landmark_3d(hand_landmarks, 2))
            i_tip = np.array(_landmark_3d(hand_landmarks, 8))
            i_mcp_v = np.array(_landmark_3d(hand_landmarks, 5))
            v_t = t_tip - t_mcp
            v_i = i_tip - i_mcp_v
            cos_abd = np.dot(v_t, v_i) / (np.linalg.norm(v_t) * np.linalg.norm(v_i) + 1e-8)
            thumb_abd = np.degrees(np.arccos(np.clip(cos_abd, -1.0, 1.0)))

            if t_mcp_3d > 155 and thumb_abd > 25:
                return 'Eight'
            return 'One'

        # Two: index + middle (all 3 variants: wide/medium/together)
        if I and M:
            return 'Two'

        return 'Two'

    if n == 3:
        # Seven (v5): thumb + index + pinky extended, middle + ring curled (2D)
        if T and I and P and not M and not R:
            return 'Seven'

        # Three: any 3 consecutive fingers
        if T and I and M:
            return 'Three'
        if I and M and R:
            return 'Three'
        if M and R and P:
            return 'Three'

        return 'Three'

    if n == 4:
        # Four: only thumb bent, other 4 extended
        if not T:
            return 'Four'
        return 'Four'

    # n == 5
    return 'Open_Palm'


def each_finger_up_mcp(hand_landmarks):
    """v2.1: MCP-based finger extension detection (more reliable than PIP).
    Uses wrist→MCP→PIP angles instead of MCP→PIP→DIP.
    Reference: Hamaguchi et al. (2024) found MCP angles have higher
    correlation with gold-standard (r=0.78 vs r=0.72 for PIP).
    """
    mcp_joint_list = [[2, 1, 0], [5, 4, 0], [9, 7, 0], [13, 10, 0], [17, 13, 0]]
    is_finger_up = []
    for a_idx, b_idx, c_idx in mcp_joint_list:
        a = np.array([hand_landmarks.landmark[a_idx].x, hand_landmarks.landmark[a_idx].y])
        b = np.array([hand_landmarks.landmark[b_idx].x, hand_landmarks.landmark[b_idx].y])
        c = np.array([hand_landmarks.landmark[c_idx].x, hand_landmarks.landmark[c_idx].y])
        angle = cal_angle(a, b, c)
        # MCP extension angle threshold is lower (150° vs 165° for PIP)
        is_finger_up.append(angle > 150)
    return is_finger_up


def compute_pip_confidence(hand_landmarks):
    """v2.1: Estimate tracking quality at PIP joints.
    PIP joints with extreme angles or unrealistic positions suggest tracking error.
    Returns 0.0-1.0 confidence where 1.0 = all PIP joints look reliable.
    """
    # PIP → corresponding MCP mapping
    pip_to_mcp = {3: 2, 6: 5, 10: 9, 14: 13, 18: 17}

    confidence = 1.0
    penalties = 0
    for pip, mcp in pip_to_mcp.items():
        pip_pos = np.array([hand_landmarks.landmark[pip].x,
                            hand_landmarks.landmark[pip].y])
        mcp_pos = np.array([hand_landmarks.landmark[mcp].x,
                            hand_landmarks.landmark[mcp].y])

        # Check: PIP-to-MCP distance within anatomically plausible range
        dist = np.linalg.norm(pip_pos - mcp_pos)
        if dist > 0.3:  # Normalized distance > 0.3 is suspicious
            penalties += 1

    confidence -= penalties * 0.12  # Each suspicious PIP reduces confidence
    return max(confidence, 0.4)      # Floor at 0.4


class RuleRecognizer:
    """Rule-based CSL 0-9 digit gesture recognizer using joint angles.
    Fallback for deep learning.

    v4.0: CSL digits only. pinch_threshold controls CSL 7 pinch sensitivity.
    v2.1: PIP error compensation with MCP auxiliary checks.
    """

    def __init__(self, pinch_threshold=0.08):
        self.pinch_threshold = pinch_threshold

    def recognize(self, hand_landmarks_wrapper):
        """Return (finger_up_list, finger_count, gesture_name)."""
        finger_up = each_finger_up(hand_landmarks_wrapper)
        finger_count = sum(finger_up)
        gesture = rec_gesture(finger_up, hand_landmarks_wrapper, self.pinch_threshold)
        return finger_up, finger_count, gesture

    def recognize_with_confidence(self, hand_landmarks_wrapper):
        """v2.1: Recognize gesture with PIP-aware confidence scoring.

        Returns: (finger_up_list, finger_count, gesture_name, confidence)
          confidence: 0.0-1.0 reflecting tracking quality and gesture reliability
        """
        finger_up = each_finger_up(hand_landmarks_wrapper)
        finger_count = sum(finger_up)
        gesture = rec_gesture(finger_up, hand_landmarks_wrapper, self.pinch_threshold)

        # Base confidence
        confidence = 1.0

        # PIP joint quality check
        pip_conf = compute_pip_confidence(hand_landmarks_wrapper)
        confidence *= pip_conf

        # Extra check for PIP-sensitive gestures: verify with MCP angles
        if gesture in PIP_SENSITIVE_GESTURES:
            finger_up_mcp = each_finger_up_mcp(hand_landmarks_wrapper)
            # Compare PIP-based vs MCP-based finger states
            agreements = sum(1 for a, b in zip(finger_up, finger_up_mcp) if a == b)
            agreement_rate = agreements / 5.0
            # Mismatch between PIP and MCP suggests tracking error
            confidence *= (0.6 + 0.4 * agreement_rate)

        return finger_up, finger_count, gesture, float(confidence)
