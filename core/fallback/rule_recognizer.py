# coding:utf-8
"""
Rule-based gesture recognition — migrated from tools.py.
Serves as fallback when deep learning model is unavailable.

v2.1: PIP joint error compensation (Wagh et al. 2025).
PIP joints (landmarks 6,10,14,18) have highest MediaPipe error.
Gestures that rely on PIP accuracy get MCP-based auxiliary checks
and confidence penalties to reflect real-world tracking uncertainty.
"""
import numpy as np

# Landmark indices
MCP_JOINTS = [2, 5, 9, 13, 17]   # Metacarpophalangeal (most reliable)
PIP_JOINTS = [3, 6, 10, 14, 18]  # Proximal interphalangeal (highest error)
TIP_JOINTS = [4, 8, 12, 16, 20]  # Fingertips (moderate error)

# Gestures that depend heavily on PIP joint accuracy
PIP_SENSITIVE_GESTURES = {'Victory', 'Space', 'Six', 'Three', 'Four', 'Seven', 'Nine'}


def cal_angle(a, b, c):
    """Calculate angle at point b formed by points a-b-c."""
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


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


def get_palm_angle(hand_landmarks):
    """Compute palm angle: < 20 horizontal, > 70 vertical."""
    ring_point = (hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y)
    wrist_point = (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)
    horizon_point = (hand_landmarks.landmark[0].x + 0.5, hand_landmarks.landmark[0].y)
    angle = cal_angle(ring_point, wrist_point, horizon_point)
    if angle > 90:
        angle = 180 - angle
    return angle


def rec_gesture(is_finger_up, hand_landmarks):
    """Recognize gesture from finger state and palm angle. Returns gesture name string."""
    palm_angle = get_palm_angle(hand_landmarks)

    if sum(is_finger_up) == 0:
        gesture = 'Closed_Fist'
    elif sum(is_finger_up) == 1:
        if is_finger_up[0] and palm_angle < 30:
            gesture = 'Good'
        elif is_finger_up[4] and not is_finger_up[0] and not is_finger_up[1] and not is_finger_up[2] and not is_finger_up[3]:
            gesture = 'Pinky_Up'
        else:
            gesture = 'One'
    elif sum(is_finger_up) == 2:
        if is_finger_up[1] and is_finger_up[2]:
            # Differentiate Victory (fingers apart) vs Two (fingers together)
            # by measuring distance between index tip (8) and middle tip (12)
            idx_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
            mid_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y])
            dist = np.linalg.norm(idx_tip - mid_tip)
            gesture = 'Two' if dist < 0.06 else 'Victory'
        elif is_finger_up[0] and is_finger_up[1]:
            gesture = 'Eight'
        elif is_finger_up[0] and is_finger_up[4]:
            gesture = 'Six'
        elif is_finger_up[2] and is_finger_up[3]:
            gesture = 'Mid_Ring'
        else:
            gesture = 'Two'
    elif sum(is_finger_up) == 3:
        if is_finger_up[0] and is_finger_up[1] and is_finger_up[2]:
            gesture = 'Seven'
        elif is_finger_up[0] and is_finger_up[1] and is_finger_up[4]:
            gesture = 'Space'
        else:
            gesture = 'Three'
    elif sum(is_finger_up) == 4:
        # v3.4: Nine = ring finger curled, other 4 extended
        if not is_finger_up[3]:
            gesture = 'Nine'
        else:
            gesture = 'Four'
    else:
        gesture = 'Open_Palm'
    return gesture


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
    """Rule-based gesture recognizer using joint angles. Fallback for deep learning.

    v2.1: Added PIP error compensation — uses MCP-based finger detection
    as auxiliary check for gestures that depend heavily on PIP joint accuracy.
    """

    def recognize(self, hand_landmarks_wrapper):
        """Return (finger_up_list, finger_count, gesture_name)."""
        finger_up = each_finger_up(hand_landmarks_wrapper)
        finger_count = sum(finger_up)
        gesture = rec_gesture(finger_up, hand_landmarks_wrapper)
        return finger_up, finger_count, gesture

    def recognize_with_confidence(self, hand_landmarks_wrapper):
        """v2.1: Recognize gesture with PIP-aware confidence scoring.

        Returns: (finger_up_list, finger_count, gesture_name, confidence)
          confidence: 0.0-1.0 reflecting tracking quality and gesture reliability
        """
        finger_up = each_finger_up(hand_landmarks_wrapper)
        finger_count = sum(finger_up)
        gesture = rec_gesture(finger_up, hand_landmarks_wrapper)

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
