# coding:utf-8
"""
Multi-user gesture data recorder v1.1 — low-cost generalization data collection.

Records all 16 gesture variants × 2 hands from multiple users.
Fixed condition (NORMAL light + MID distance + CENTER angle), one take each.
Output: data/multiuser_sequences/ + data/multiuser_annotations.json

Usage:
  python tools/record_multiuser.py --user alice
  python tools/record_multiuser.py --user bob
"""

import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from collections import Counter

from core.perception.hand_tracker import HandTracker, HandLandmarksWrapper
from core.fallback.rule_recognizer import RuleRecognizer
from core.perception.flow_estimator import FlowEstimator
from core.feature.motion_encoder import compute_flow_histogram

# All 16 variants (same as record_data.py v5.3)
GESTURE_VARIANTS = [
    ('Closed_Fist', 'front_fist',       'Clench all 5 fingers tight'),
    ('One',         'index_front',       'Only INDEX finger up, front view'),
    ('One',         'thumb_front',       'Only THUMB up, front view'),
    ('Two',         'wide',              'Index+middle WIDE apart (V shape)'),
    ('Two',         'medium',            'Index+middle MEDIUM spread'),
    ('Two',         'together',          'Index+middle TOGETHER (no gap)'),
    ('Three',       'T_I_M',             'Thumb+index+middle up, ring+pinky curled'),
    ('Three',       'I_M_R',             'Index+middle+ring up, thumb+pinky curled'),
    ('Three',       'M_R_P',             'Middle+ring+pinky up, thumb+index curled'),
    ('Four',        'four_fingers',      'Only thumb bent, other 4 extended'),
    ('Open_Palm',   'open_palm',         'All 5 fingers spread wide open'),
    ('Six',         'thumb_pinky',       'Thumb+pinky out, middle 3 clenched'),
    ('Seven',       'thumb_index_pinky', 'Thumb+index+pinky UP, middle+ring CURLED'),
    ('Eight',       'L_shape',           'Thumb+index L-shape, thumb abducted'),
    ('Nine',        'front',             'Hooked index finger, FRONT view'),
    ('Nine',        'side',              'Hooked index finger, SIDE view'),
]

RECORD_SECONDS = 3.0
MIN_FRAMES = 12


def wait_key(cap, prompt_lines):
    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        overlay = frame.copy()
        panel_h = 18 + 22 * len(prompt_lines)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_h), (30, 30, 30), -1)
        frame[:] = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        for i, (text, color) in enumerate(prompt_lines):
            cv2.putText(frame, text, (10, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('Multi-User Recorder', frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 32: return 'record'
        if k == ord('s'): return 'skip'
        if k == ord('q') or k == 27: return 'quit'


def record_sequence(cap, tracker, flow_estimator, recognizer, data_dir,
                    user_id, hand, gesture_name, gesture_hint):
    """Record one 3-second sequence with quality check. Retries up to 3 times."""
    for attempt in range(3):
        result = _record_one_take(cap, tracker, flow_estimator, recognizer,
                                  data_dir, user_id, hand, gesture_name)
        if result == 'retry':
            print(f'  Retry ({attempt+1}/3) — hold gesture steady')
            for _ in range(30):
                cv2.waitKey(1)
            continue
        return result
    print(f'  Gave up after 3 attempts')
    return None


def _record_one_take(cap, tracker, flow_estimator, recognizer,
                     data_dir, user_id, hand, gesture_name):
    frames, roi_frames, flow_hists = [], [], []
    recognized = []
    t0 = time.time()
    flow_estimator.reset()
    frame_interval = 1.0 / 25.0
    last_t = 0

    while time.time() - t0 < RECORD_SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if now - last_t < frame_interval:
            cv2.waitKey(1)
            continue
        last_t = now

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = tracker.detect_video(rgb, int(now * 1000))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_field = flow_estimator.compute(gray)

        r = RECORD_SECONDS - (now - t0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 160), -1)
        frame[:] = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        rec_color = (0, 255, 0) if (recognized and recognized[-1] == gesture_name) else (0, 140, 255)
        rec_text = recognized[-1] if recognized else '?'
        cv2.putText(frame, f'REC {r:.1f}s | {gesture_name} | {hand} | rule={rec_text}',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 2)

        if result.hand_landmarks:
            w0 = HandLandmarksWrapper(result.hand_landmarks[0])
            kp = np.array([[lm.x, lm.y, lm.z] for lm in w0.landmark], dtype=np.float32)
            frames.append(kp)
            _, _, rec_ges = recognizer.recognize(w0)
            recognized.append(rec_ges)

            xs = [int(lm.x * w) for lm in w0.landmark]
            ys = [int(lm.y * h) for lm in w0.landmark]
            pad = 25
            x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
            x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
            if x2 > x1 and y2 > y1:
                roi = cv2.resize(frame[y1:y2, x1:x2], (96, 96))
            else:
                roi = np.zeros((96, 96, 3), dtype=np.uint8)
            roi_frames.append(roi)

            flow_crop = flow_field[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else flow_field
            flow_hist = compute_flow_histogram(flow_crop, bins=64)
            flow_hists.append(flow_hist)

            for lm in w0.landmark:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

        cv2.imshow('Multi-User Recorder', frame)
        cv2.waitKey(1)

    for _ in range(15):
        cv2.waitKey(1)

    if len(frames) < MIN_FRAMES:
        print(f'  SKIP ({len(frames)}f < {MIN_FRAMES})')
        return None

    if recognized:
        correct = sum(1 for r in recognized if r == gesture_name)
        rate = correct / len(recognized)
        top = Counter(recognized).most_common(3)
        top_str = ' | '.join(f'{g}:{c}' for g, c in top)
        if rate < 0.4:
            print(f'  QUALITY FAIL ({rate:.0%} correct, saw: {top_str})')
            return 'retry'
        print(f'  OK ({rate:.0%} match, {top_str})')

    arr = np.array(frames, dtype=np.float32)
    roi_arr = np.array(roi_frames, dtype=np.uint8)
    flow_arr = np.array(flow_hists, dtype=np.float32)

    base = f'{user_id}_{hand}_{gesture_name}'
    kp_path = os.path.join(data_dir, f'{base}.npy')
    roi_path = os.path.join(data_dir, f'{base}_roi.npy')
    flow_path = os.path.join(data_dir, f'{base}_flow.npy')

    np.save(kp_path, arr)
    np.save(roi_path, roi_arr)
    np.save(flow_path, flow_arr)

    print(f'  SAVED ({len(frames)}f) -> {base}')
    return {
        'video_id': base,
        'label': gesture_name,
        'user_id': user_id,
        'hand': hand,
        'variant': f'{user_id}_{hand}_{gesture_name}_NORMAL_MID_CENTER',
        'frames_path': kp_path,
        'roi_path': roi_path,
        'flow_path': flow_path,
        'num_frames': len(frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', required=True, help='User identifier (e.g. alice, bob)')
    parser.add_argument('--output', default='data/multiuser_annotations.json')
    parser.add_argument('--data_dir', default='data/multiuser_sequences')
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    tracker = HandTracker(min_detection_confidence=0.35, min_presence_confidence=0.35,
                          min_tracking_confidence=0.25)
    recognizer = RuleRecognizer()
    flow_estimator = FlowEstimator()
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    total = len(GESTURE_VARIANTS) * 2  # 16 variants × 2 hands
    annotations = []

    print(f'  Multi-User Recorder v1.1 — User: {args.user}')
    print(f'  {len(GESTURE_VARIANTS)} variants × 2 hands = {total} samples')
    print(f'  3s per sample | SPACE=record S=skip Q=quit')
    print()

    for hand in ['Right', 'Left']:
        print(f'\n{"="*60}')
        print(f'  >>> HAND: {hand} — use your {hand.upper()} hand')
        print(f'{"="*60}')
        for gesture_name, var_id, gesture_hint in GESTURE_VARIANTS:
            sample_idx = len(annotations) + 1
            prompt = [
                (f'[{sample_idx}/{total}] {args.user} | {hand} | {gesture_name}/{var_id}', (0, 255, 0)),
                (f'{gesture_hint}  |  SPACE=rec  S=skip  Q=quit', (200, 200, 200)),
            ]
            action = wait_key(cap, prompt)
            if action == 'quit':
                cap.release()
                cv2.destroyAllWindows()
                tracker.close()
                _save(annotations, args.output)
                return
            if action == 'skip':
                continue

            ann = record_sequence(cap, tracker, flow_estimator, recognizer,
                                  args.data_dir, args.user, hand, gesture_name, gesture_hint)
            if isinstance(ann, dict):
                annotations.append(ann)

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    _save(annotations, args.output)


def _save(annotations, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    labels = Counter(a['label'] for a in annotations)
    users = Counter(a['user_id'] for a in annotations)
    print(f'\nDone! {len(annotations)} samples -> {output_path}')
    print(f'Users: {dict(users)}')
    print(f'Labels: {dict(sorted(labels.items()))}')


if __name__ == '__main__':
    main()
