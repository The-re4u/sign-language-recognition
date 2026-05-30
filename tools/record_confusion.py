# coding:utf-8
"""Confusion-pair hard sample recorder v2.0.

Records edge-case gestures at classification boundaries for model evaluation.
6 confusion groups x 6 takes each + 12 edge extras = ~48 samples.

Usage:
  python tools/record_confusion.py --output data/confusion_v6.json --data_dir data/confusion_v6
"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from collections import Counter
from core.perception.hand_tracker import HandTracker, HandLandmarksWrapper
from core.fallback.rule_recognizer import RuleRecognizer

# ── Confusion pairs v2.0 ──
# (gesture_name, variant_id, hint_en)
# hint_en appears in camera overlay and terminal — keep it short ASCII
CONFUSION_PAIRS = [
    # Group 1: Three <-> Two  (ring finger PIP at boundary ~155-165 deg)
    ('Two',   'to_Three',  'Two->Three: ring finger slightly extended, PIP~155deg'),
    ('Two',   'to_Three',  'Two->Three: ring finger slightly extended, PIP~160deg'),
    ('Two',   'to_Three',  'Two->Three: ring finger slightly extended, PIP~150deg'),
    ('Three', 'to_Two',    'Three->Two: ring finger slightly bent, PIP~160deg'),
    ('Three', 'to_Two',    'Three->Two: ring finger slightly bent, PIP~155deg'),
    ('Three', 'to_Two',    'Three->Two: ring finger slightly bent, PIP~165deg'),

    # Group 2: Four <-> Open_Palm  (thumb angle at boundary)
    ('Four',      'to_Palm', 'Four->Open: thumb slightly extended, IP~160deg'),
    ('Four',      'to_Palm', 'Four->Open: thumb slightly extended, IP~155deg'),
    ('Four',      'to_Palm', 'Four->Open: thumb slightly extended, IP~150deg'),
    ('Open_Palm', 'to_Four', 'Open->Four: thumb slightly bent, IP~160deg'),
    ('Open_Palm', 'to_Four', 'Open->Four: thumb slightly bent, IP~155deg'),
    ('Open_Palm', 'to_Four', 'Open->Four: thumb slightly bent, IP~150deg'),

    # Group 3: One <-> Nine  (index PIP at hook boundary)
    ('One',  'to_Nine', 'One->Nine: index PIP slightly bent, ~155deg'),
    ('One',  'to_Nine', 'One->Nine: index PIP slightly bent, ~160deg'),
    ('One',  'to_Nine', 'One->Nine: index PIP slightly bent, ~150deg'),
    ('Nine', 'to_One',  'Nine->One: index PIP slightly straightened, ~160deg'),
    ('Nine', 'to_One',  'Nine->One: index PIP slightly straightened, ~155deg'),
    ('Nine', 'to_One',  'Nine->One: index PIP slightly straightened, ~165deg'),

    # Group 4: Six <-> Seven  (index finger partial extend)
    ('Six',   'to_Seven', 'Six->Seven: index slightly extended, PIP~140deg'),
    ('Six',   'to_Seven', 'Six->Seven: index slightly extended, PIP~150deg'),
    ('Six',   'to_Seven', 'Six->Seven: index slightly extended, PIP~130deg'),
    ('Seven', 'to_Six',   'Seven->Six: index slightly bent, PIP~150deg'),
    ('Seven', 'to_Six',   'Seven->Six: index slightly bent, PIP~140deg'),
    ('Seven', 'to_Six',   'Seven->Six: index slightly bent, PIP~160deg'),

    # Group 5: Seven <-> Eight  (pinky partial)
    ('Seven', 'to_Eight', 'Seven->Eight: pinky slightly bent, PIP~150deg'),
    ('Seven', 'to_Eight', 'Seven->Eight: pinky slightly bent, PIP~140deg'),
    ('Seven', 'to_Eight', 'Seven->Eight: pinky slightly bent, PIP~160deg'),
    ('Eight', 'to_Seven', 'Eight->Seven: pinky slightly extended, PIP~150deg'),
    ('Eight', 'to_Seven', 'Eight->Seven: pinky slightly extended, PIP~140deg'),
    ('Eight', 'to_Seven', 'Eight->Seven: pinky slightly extended, PIP~160deg'),

    # Group 6: Closed_Fist <-> Open_Palm  (half-open fist)
    ('Closed_Fist', 'half_open', 'Half fist: fingers bent ~120deg MCP'),
    ('Closed_Fist', 'half_open', 'Half fist: fingers bent ~140deg MCP'),
    ('Closed_Fist', 'half_open', 'Half fist: fingers bent ~100deg MCP'),
    ('Open_Palm',   'half_close','Half open: fingers slightly curved, MCP~150deg'),
    ('Open_Palm',   'half_close','Half open: fingers slightly curved, MCP~140deg'),
    ('Open_Palm',   'half_close','Half open: fingers slightly curved, MCP~160deg'),

    # Edge extras: different angles/distances for most confused pairs
    ('Three', 'to_Two_angle', 'Three->Two: wrist rotated ~30deg left, ring bent'),
    ('Three', 'to_Two_angle', 'Three->Two: wrist rotated ~30deg right, ring bent'),
    ('Two',   'to_Three_far', 'Two->Three: hand farther from camera, ring extended'),
    ('Two',   'to_Three_far', 'Two->Three: hand closer to camera, ring extended'),
    ('Four',  'to_Palm_angle','Four->Open: wrist rotated ~30deg, thumb extended'),
    ('Four',  'to_Palm_angle','Four->Open: wrist rotated ~30deg, thumb bent'),
    ('One',   'to_Nine_angle','One->Nine: hand turned ~45deg, PIP boundary'),
    ('One',   'to_Nine_angle','One->Nine: hand farther, PIP boundary'),
    ('Nine',  'to_One_angle', 'Nine->One: side view, PIP slightly straight'),
    ('Nine',  'to_One_angle', 'Nine->One: 45-degree angle, PIP boundary'),
    ('Open_Palm','half_close_angle','Half open: wrist rotated, MCP boundary'),
    ('Closed_Fist','half_open_angle','Half fist: wrist rotated, MCP boundary'),
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
        panel_h = 18 + 24 * len(prompt_lines)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_h), (30, 30, 30), -1)
        frame[:] = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        for i, (text, color) in enumerate(prompt_lines):
            cv2.putText(frame, text, (10, 22 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.imshow('Confusion Recorder v2', frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 32: return 'record'
        if k == ord('s'): return 'skip'
        if k == ord('q') or k == 27: return 'quit'


def _crop_roi(frame, kp_xy, h, w, size=96, pad=25):
    xs = [int(x * w) for x in kp_xy[:, 0]]
    ys = [int(y * h) for y in kp_xy[:, 1]]
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
    if x2 > x1 and y2 > y1:
        return cv2.resize(frame[y1:y2, x1:x2], (size, size))
    return np.zeros((size, size, 3), dtype=np.uint8)


def _record_one_take(cap, tracker, recognizer, data_dir, gesture_name, var_id, hint):
    frames, roi_frames = [], []
    recognized = []
    t0 = time.time()
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

        r = RECORD_SECONDS - (now - t0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 160), -1)
        frame[:] = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        rec_color = (0, 255, 0) if (recognized and recognized[-1] == gesture_name) else (0, 140, 255)
        rec_text = recognized[-1] if recognized else '?'
        cv2.putText(frame, f'REC {r:.1f}s | {gesture_name}/{var_id} | rule={rec_text}',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 1)

        if result.hand_landmarks:
            w0 = HandLandmarksWrapper(result.hand_landmarks[0])
            kp = np.array([[lm.x, lm.y, lm.z] for lm in w0.landmark], dtype=np.float32)
            frames.append(kp)
            roi_frames.append(_crop_roi(frame, kp[:, :2], h, w))
            _, _, rec_ges = recognizer.recognize(w0)
            recognized.append(rec_ges)

            for lm in w0.landmark:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

        cv2.imshow('Confusion Recorder v2', frame)
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
        if rate < 0.2:  # very low bar: these are edge cases
            print(f'  QUALITY FAIL ({rate:.0%}, saw: {top_str})')
            return 'retry'
        print(f'  OK ({rate:.0%} match, {top_str})')

    arr = np.array(frames, dtype=np.float32)
    roi_arr = np.array(roi_frames, dtype=np.uint8)
    idx = int(time.time() * 1000) % 100000
    base = f'conf_{gesture_name}_{var_id}_{idx:05d}'
    kp_path = os.path.join(data_dir, f'{base}.npy')
    roi_path = os.path.join(data_dir, f'{base}_roi.npy')
    np.save(kp_path, arr)
    np.save(roi_path, roi_arr)

    print(f'  SAVED ({len(frames)}f) -> {base}')
    return {
        'video_id': base, 'label': gesture_name,
        'variant': f'conf_{gesture_name}_{var_id}_{idx:05d}',
        'confusion_pair': True, 'frames_path': kp_path, 'roi_path': roi_path,
        'num_frames': len(frames), 'hand': 'Right',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/confusion_v6.json')
    parser.add_argument('--data_dir', default='data/confusion_v6')
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    tracker = HandTracker(min_detection_confidence=0.35, min_presence_confidence=0.35,
                          min_tracking_confidence=0.25)
    recognizer = RuleRecognizer()
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    total = len(CONFUSION_PAIRS)
    annotations = []

    print(f'Confusion Recorder v2.0 — {total} samples')
    print(f'SPACE=record  S=skip  Q=quit')
    print(f'Quality threshold: 20% (these are edge cases)')
    print()

    for i, (gesture_name, var_id, hint) in enumerate(CONFUSION_PAIRS):
        print(f'\n[{i+1}/{total}] {gesture_name}/{var_id}')
        print(f'  Hint: {hint}')
        print(f'  SPACE=rec  S=skip  Q=quit')

        prompt = [
            (f'[{i+1}/{total}] {gesture_name}/{var_id}', (0, 255, 255)),
            (f'{hint[:60]}', (200, 200, 200)),
            (f'SPACE=rec  S=skip  Q=quit', (0, 255, 0)),
        ]
        action = wait_key(cap, prompt)
        if action == 'quit':
            cap.release(); cv2.destroyAllWindows(); tracker.close()
            _save(annotations, args.output); return
        if action == 'skip':
            continue

        for attempt in range(3):
            result = _record_one_take(cap, tracker, recognizer,
                                       args.data_dir, gesture_name, var_id, hint)
            if result == 'retry':
                print(f'  Retry ({attempt+1}/3)')
                for _ in range(30):
                    cv2.waitKey(1)
                continue
            if isinstance(result, dict):
                annotations.append(result)
            break
        else:
            print(f'  Gave up after 3 attempts')

    cap.release(); cv2.destroyAllWindows(); tracker.close()
    _save(annotations, args.output)


def _save(annotations, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    labels = Counter(a['label'] for a in annotations)
    print(f'\nDone! {len(annotations)} -> {output_path}')
    for lbl, cnt in sorted(labels.items()):
        print(f'  {lbl}: {cnt}')


if __name__ == '__main__':
    main()
