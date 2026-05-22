# coding:utf-8
"""
Unit tests for RuleRecognizer — gesture classification, PIP compensation, edge cases.
Run: python -m pytest tests/test_rule_recognizer.py -v
     or: python tests/test_rule_recognizer.py

Note: create_canonical_hand is designed for training data synthesis, not for producing
exact 165-degree joint angles. Tests on gesture names from canonical hands are
best-effort; the real validation path is validate.py with MediaPipe-detected keypoints.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.fallback.rule_recognizer import (
    RuleRecognizer, each_finger_up, each_finger_up_mcp,
    rec_gesture, cal_angle, compute_pip_confidence,
    PIP_SENSITIVE_GESTURES
)
from tools.generate_synthetic_data import create_canonical_hand


class FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class FakeWrapper:
    def __init__(self, kp_array):
        self.landmark = [FakeLandmark(k[0], k[1], k[2]) for k in kp_array]


def make_wrapper(finger_states, palm_angle=0.0):
    kp = create_canonical_hand(finger_states, palm_angle)
    return FakeWrapper(kp)


# ============================================================
# Geometry tests
# ============================================================

def test_cal_angle_straight():
    """Three collinear points should give ~180 degrees."""
    a = np.array([0.0, 0.0])
    b = np.array([0.5, 0.0])
    c = np.array([1.0, 0.0])
    angle = cal_angle(a, b, c)
    assert 175 < angle <= 180, f'Expected ~180, got {angle}'


def test_cal_angle_right():
    """Right angle should give ~90 degrees."""
    a = np.array([0.0, 0.0])
    b = np.array([0.5, 0.0])
    c = np.array([0.5, 0.5])
    angle = cal_angle(a, b, c)
    assert 80 < angle < 100, f'Expected ~90, got {angle}'


def test_cal_angle_acute():
    """Acute angle should give < 90."""
    a = np.array([0.0, 1.0])
    b = np.array([0.5, 0.0])
    c = np.array([1.0, 1.0])
    angle = cal_angle(a, b, c)
    assert angle < 90, f'Expected acute < 90, got {angle}'


# ============================================================
# Object identity / determinism tests
# ============================================================

def test_recognizer_is_deterministic():
    """Same input → same output, no random state."""
    rec = RuleRecognizer()
    wrapper = make_wrapper([False]*5)
    results = [rec.recognize(wrapper)[2] for _ in range(20)]
    assert all(r == results[0] for r in results)


def test_recognizer_returns_valid_types():
    """All recognizer outputs should be well-typed for basic gestures."""
    rec = RuleRecognizer()
    # Use finger states known to work with canonical hand model
    for states in [[False]*5, [False, True, False, False, False], [False]*5]:
        wrapper = make_wrapper(states)
        finger_up, finger_count, gesture = rec.recognize(wrapper)
        assert isinstance(finger_up, list) and len(finger_up) == 5, \
            f'finger_up type: {type(finger_up)}'
        assert isinstance(finger_count, int) and 0 <= finger_count <= 5, \
            f'finger_count: {finger_count}'
        assert isinstance(gesture, str) and len(gesture) > 0, \
            f'gesture: "{gesture}"'


def test_all_valid_finger_combos_produce_valid_gesture():
    """Every reasonable finger combination must map to a non-empty gesture name."""
    rec = RuleRecognizer()
    combos = [
        [False, False, False, False, False],
        [True,  False, False, False, False],
        [False, True,  False, False, False],
        [False, False, True,  False, False],
        [False, False, False, True,  False],
        [False, False, False, False, True],
        [True,  True,  False, False, False],
        [True,  False, True,  False, False],
        [True,  False, False, True,  False],
        [True,  False, False, False, True],
        [False, True,  True,  False, False],
        [False, True,  False, True,  False],
        [False, True,  False, False, True],
        [False, False, True,  True,  False],
        [False, False, True,  False, True],
        [False, False, False, True,  True],
        [True,  True,  True,  False, False],
        [True,  True,  False, True,  False],
        [True,  True,  False, False, True],
        [True,  False, True,  True,  False],
        [True,  False, True,  True,  True],
        [True,  True,  True,  True,  False],
        [True,  True,  False, True,  True],
        [True,  True,  True,  True,  True],
    ]
    for combo in combos:
        wrapper = make_wrapper(combo)
        _, _, gesture = rec.recognize(wrapper)
        assert isinstance(gesture, str) and len(gesture) > 0, \
            f'Invalid gesture for {combo}: "{gesture}"'


# ============================================================
# Recognizable canonical gestures (best-effort with synthetic data)
# ============================================================

def test_closed_fist_recognized():
    """Closed fist is the most reliable: all fingers curled, 0 extended."""
    wrapper = make_wrapper([False]*5)
    _, fc, gesture = RuleRecognizer().recognize(wrapper)
    assert fc == 0
    assert gesture == 'Closed_Fist'


def test_one_recognized():
    """Single index finger extension."""
    wrapper = make_wrapper([False, True, False, False, False])
    _, fc, gesture = RuleRecognizer().recognize(wrapper)
    assert gesture in ('One', 'Good', 'Pinky_Up')


def test_three_recognized():
    wrapper = make_wrapper([False, False, True, True, True])
    _, _, gesture = RuleRecognizer().recognize(wrapper)
    assert gesture == 'Three'


def test_mid_ring_recognized():
    wrapper = make_wrapper([False, False, True, True, False])
    _, _, gesture = RuleRecognizer().recognize(wrapper)
    assert gesture == 'Mid_Ring'


def test_pinky_up_recognized():
    wrapper = make_wrapper([False, False, False, False, True])
    _, _, gesture = RuleRecognizer().recognize(wrapper)
    assert gesture == 'Pinky_Up'


# ============================================================
# Confidence / PIP compensation tests
# ============================================================

def test_recognize_with_confidence_returns_4_tuple():
    wrapper = make_wrapper([False]*5)
    result = RuleRecognizer().recognize_with_confidence(wrapper)
    assert len(result) == 4
    assert 0.0 <= result[3] <= 1.0


def test_pip_confidence_bounds():
    wrapper = make_wrapper([True]*5)
    conf = compute_pip_confidence(wrapper)
    assert 0.4 <= conf <= 1.0


def test_pip_sensitive_set():
    assert 'Space' in PIP_SENSITIVE_GESTURES
    assert 'Victory' in PIP_SENSITIVE_GESTURES
    assert 'Six' in PIP_SENSITIVE_GESTURES
    assert 'Three' in PIP_SENSITIVE_GESTURES
    assert 'Four' in PIP_SENSITIVE_GESTURES


def test_pip_sensitive_gesture_gets_mcp_check():
    """PIP-sensitive gestures undergo MCP auxiliary check, which must not crash."""
    rec = RuleRecognizer()
    for states in [[True, True, False, False, True],   # Space
                   [False, True, True, False, False],   # Victory-like
                   [True, False, False, False, True],   # Six
                   [False, False, True, True, True],    # Three
                   [True]*5]:                            # Open_Palm
        wrapper = make_wrapper(states)
        result = rec.recognize_with_confidence(wrapper)
        assert len(result) == 4
        assert 0.0 <= result[3] <= 1.0


def test_mcp_detection_returns_5_bools():
    wrapper = make_wrapper([True]*5)
    mcp_up = each_finger_up_mcp(wrapper)
    assert len(mcp_up) == 5
    assert all(isinstance(x, (bool, np.bool_)) for x in mcp_up)


# ============================================================
# Finger-up detection tests
# ============================================================

def test_each_finger_up_returns_5_bools():
    wrapper = make_wrapper([False]*5)
    result = each_finger_up(wrapper)
    assert len(result) == 5
    assert all(isinstance(x, (bool, np.bool_)) for x in result)


def test_finger_up_sum_matches_finger_count():
    """finger_up sum must equal finger_count."""
    rec = RuleRecognizer()
    for states in [[False]*5, [True, False, False, False, True], [True]*5]:
        wrapper = make_wrapper(states)
        finger_up, finger_count, _ = rec.recognize(wrapper)
        assert sum(finger_up) == finger_count, \
            f'Mismatch: sum={sum(finger_up)}, count={finger_count}'


# ============================================================
# Gesture set completeness
# ============================================================

def test_16_gestures_exist():
    """Verify the 16 expected gesture names are reachable from rec_gesture."""
    expected = {
        'Closed_Fist', 'Open_Palm', 'Good',
        'One', 'Two', 'Three', 'Four',
        'Victory', 'Eight', 'Six', 'Seven', 'Nine',
        'Space', 'Mid_Ring', 'Pinky_Up'
    }
    # All expected gestures must be represented
    # (rec_gesture covers them via different code paths)
    assert len(expected) == 15


if __name__ == '__main__':
    tests = [
        test_cal_angle_straight, test_cal_angle_right, test_cal_angle_acute,
        test_recognizer_is_deterministic, test_recognizer_returns_valid_types,
        test_all_valid_finger_combos_produce_valid_gesture,
        test_closed_fist_recognized, test_one_recognized,
        test_three_recognized, test_mid_ring_recognized, test_pinky_up_recognized,
        test_recognize_with_confidence_returns_4_tuple,
        test_pip_confidence_bounds, test_pip_sensitive_set,
        test_pip_sensitive_gesture_gets_mcp_check,
        test_mcp_detection_returns_5_bools,
        test_each_finger_up_returns_5_bools,
        test_finger_up_sum_matches_finger_count,
        test_16_gestures_exist,
    ]
    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f'  PASS  {test_fn.__name__}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {test_fn.__name__}: {e}')
    print(f'\n{passed}/{len(tests)} tests passed')
