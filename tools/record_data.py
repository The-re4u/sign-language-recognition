# coding:utf-8
"""
Gesture data recorder v6.0 — 10 CSL digit gestures, 15 variants (Nine side only).

Single-hand mode: --hand Left/Right
  15 variants x 3 lights x 3 distances x 3 angles = 405 samples
  Saves: {base}.npy  ([T, 21, 3])

Bimanual mode: --hand Both
  15 variants x 1 condition (NORMAL/MID/CENTER)
  Saves: {base}.npy  ([T, 2, 21, 3])

Usage:
  python tools/record_data.py --output data/my_annotations.json --data_dir data/my_sequences
  python tools/record_data.py --output data/bimanual.json --data_dir data/bimanual --hand Both
"""

import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from core.perception.hand_tracker import HandTracker, HandLandmarksWrapper
from core.fallback.rule_recognizer import RuleRecognizer

# ============================================================
# Gesture definitions — 10 gestures, 18 recording variants
# ============================================================

GESTURE_VARIANTS = {
    'Closed_Fist': [
        ('front_fist', 'Clench all 5 fingers tight (front view)'),
    ],
    'One': [
        ('index_front',  'Only INDEX finger up, front view'),
        ('thumb_front',  'Only THUMB up, front view'),
    ],
    'Two': [
        ('wide',     'Index+middle WIDE apart (V shape)'),
        ('medium',   'Index+middle MEDIUM spread'),
        ('together', 'Index+middle TOGETHER (no gap)'),
    ],
    'Three': [
        ('T_I_M',  'Thumb+index+middle up, ring+pinky curled'),
        ('I_M_R',  'Index+middle+ring up, thumb+pinky curled'),
        ('M_R_P',  'Middle+ring+pinky up, thumb+index curled'),
    ],
    'Four': [
        ('four_fingers', 'Only thumb bent, other 4 fingers extended'),
    ],
    'Open_Palm': [
        ('open_palm', 'All 5 fingers spread wide open'),
    ],
    'Six': [
        ('thumb_pinky', 'Thumb+pinky out, other 3 clenched'),
    ],
    'Seven': [
        ('thumb_index_pinky', 'Thumb+index+pinky UP, middle+ring CURLED'),
    ],
    'Eight': [
        ('L_shape', 'Thumb+index L-shape, thumb abducted sideways'),
    ],
    'Nine': [
        ('side',  'Hooked index finger, SIDE view (bend visible)'),
    ],
}

# Condition levels
LIGHTS = [
    ('BRIGHT', 'BRIGHT'),
    ('NORMAL', 'NORMAL'),
    ('DIM',    'DIM'),
]
DISTS = ['NEAR', 'MID', 'FAR']
ANGLES = ['LEFT', 'CENTER', 'RIGHT']

RECORD_SECONDS = 3.0
MIN_FRAMES = 12
OUTPUT_DIR_DEFAULT = 'data/my_sequences'


def wait_key(cap, prompt_lines):
    """Show prompt and wait for SPACE/S/Q."""
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
        cv2.imshow('Recorder v5.2', frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 32:          return 'record'
        if k == ord('s'):    return 'skip'
        if k == ord('q') or k == 27: return 'quit'


def record_sequence(cap, tracker, recognizer, args, label, gesture_name, hand='Right'):
    """Record one gesture sequence. Auto-retries if gesture drifts.

    Returns annotation dict, 'retry', or None.
    """
    max_attempts = 3
    for attempt in range(max_attempts):
        result = _record_one_take(cap, tracker, recognizer, args, label, gesture_name, hand)
        if result == 'retry':
            print(f'  RETRY ({attempt+1}/{max_attempts}) — gesture changed, redo this condition')
            for _ in range(30):
                cv2.waitKey(1)
            continue
        return result
    print(f'  GAVE UP after {max_attempts} attempts')
    return None


def _crop_roi(frame, kp_xy, h, w, size=96, pad=25):
    """Crop hand ROI from frame. Returns (96, 96, 3) uint8 array or zeros."""
    xs = [int(x * w) for x in kp_xy[:, 0]]
    ys = [int(y * h) for y in kp_xy[:, 1]]
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
    if x2 > x1 and y2 > y1:
        return cv2.resize(frame[y1:y2, x1:x2], (size, size))
    return np.zeros((size, size, 3), dtype=np.uint8)


def _record_one_take(cap, tracker, recognizer, args, label, gesture_name, hand='Right'):
    """Single recording take. Bimanual (hand='Both') saves [T, 2, 21, 3]."""
    frames, roi_frames = [], []
    rec_left, rec_right = [], []
    t0 = time.time()
    frame_interval = 1.0 / 25.0
    last_t = 0
    bm = (hand == 'Both')

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
        cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 160), -1)
        frame[:] = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
        rec_color = (0, 255, 0) if (rec_right and rec_right[-1] == gesture_name) else (0, 140, 255)
        rec_text = rec_right[-1] if rec_right else '?'

        if result.hand_landmarks:
            if bm and len(result.hand_landmarks) >= 2:
                raw = []
                for i in range(min(len(result.hand_landmarks), 2)):
                    w_i = HandLandmarksWrapper(result.hand_landmarks[i])
                    h_label = result.handedness[i][0].category_name
                    kp_i = np.array([[lm.x, lm.y, lm.z] for lm in w_i.landmark], dtype=np.float32)
                    _, _, ges = recognizer.recognize(w_i)
                    raw.append((h_label, kp_i, ges))
                raw.sort(key=lambda x: 0 if x[0] == 'Left' else 1)
                frames.append(np.stack([raw[0][1], raw[1][1]]))
                roi_L = _crop_roi(frame, raw[0][1][:, :2], h, w)
                roi_R = _crop_roi(frame, raw[1][1][:, :2], h, w)
                roi_frames.append(np.stack([roi_L, roi_R]))  # [2, 96, 96, 3]
                rec_left.append(raw[0][2]); rec_right.append(raw[1][2])

                for _, kp_i, _ in raw:
                    for pt in kp_i[:, :2]:
                        cv2.circle(frame, (int(pt[0]*w), int(pt[1]*h)), 2, (0, 255, 0), -1)
                l_ok = rec_left[-1] == gesture_name; r_ok = rec_right[-1] == gesture_name
                rec_color = (0, 255, 0) if (l_ok and r_ok) else (0, 140, 255)
                cv2.putText(frame, f'REC {r:.1f}s | {gesture_name} | Both | L={rec_left[-1]} R={rec_right[-1]}',
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 2)
            elif bm:
                # Bimanual mode but only 1 hand detected — pad with zeros for missing hand
                w0 = HandLandmarksWrapper(result.hand_landmarks[0])
                h_label = result.handedness[0][0].category_name
                kp = np.array([[lm.x, lm.y, lm.z] for lm in w0.landmark], dtype=np.float32)
                zero_kp = np.zeros_like(kp)
                _, _, ges = recognizer.recognize(w0)
                if h_label == 'Left':
                    frames.append(np.stack([kp, zero_kp]))
                    roi_frames.append(np.stack([_crop_roi(frame, kp[:, :2], h, w),
                                                 np.zeros((96, 96, 3), dtype=np.uint8)]))
                    rec_left.append(ges); rec_right.append('?')
                else:
                    frames.append(np.stack([zero_kp, kp]))
                    roi_frames.append(np.stack([np.zeros((96, 96, 3), dtype=np.uint8),
                                                 _crop_roi(frame, kp[:, :2], h, w)]))
                    rec_left.append('?'); rec_right.append(ges)
                cv2.putText(frame, f'REC {r:.1f}s | {gesture_name} | Both(1h) | {h_label}={ges}',
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 2)
            else:
                w0 = HandLandmarksWrapper(result.hand_landmarks[0])
                kp = np.array([[lm.x, lm.y, lm.z] for lm in w0.landmark], dtype=np.float32)
                frames.append(kp)
                roi_frames.append(_crop_roi(frame, kp[:, :2], h, w))
                _, _, rec_ges = recognizer.recognize(w0)
                rec_right.append(rec_ges)
                cv2.putText(frame, f'REC {r:.1f}s | {gesture_name} | {hand} | rule={rec_ges}',
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 2)
                for lm in w0.landmark:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

        cv2.imshow('Recorder v5.2', frame)
        cv2.waitKey(1)

    for _ in range(15):
        cv2.waitKey(1)

    if len(frames) < MIN_FRAMES:
        print(f'  SKIP ({len(frames)}f < {MIN_FRAMES})')
        return None

    # Quality check: in bimanual mode, at least one hand must match
    if rec_right:
        if bm:
            correct = sum(1 for l, r in zip(rec_left, rec_right)
                         if l == gesture_name or r == gesture_name)
            total = len(rec_right)
        else:
            correct = sum(1 for r in rec_right if r == gesture_name)
            total = len(rec_right)
        rate = correct / max(total, 1)
        from collections import Counter
        top_all = Counter(rec_left + rec_right) if bm else Counter(rec_right)
        top = top_all.most_common(3)
        top_str = ' | '.join(f'{g}:{c}' for g, c in top)
        if rate < 0.4:
            print(f'  QUALITY FAIL ({rate:.0%}, saw: {top_str})')
            return 'retry'
        print(f'  quality: {rate:.0%} match ({top_str})')
    else:
        rate = 0

    arr = np.array(frames, dtype=np.float32)
    roi_arr = np.array(roi_frames, dtype=np.uint8)
    base = label.replace(' ', '_')
    kp_path = os.path.join(args.data_dir, f'{base}.npy')
    roi_path = os.path.join(args.data_dir, f'{base}_roi.npy')
    np.save(kp_path, arr)
    np.save(roi_path, roi_arr)

    print(f'  SAVED ({len(frames)}f, kp={arr.shape}, roi={roi_arr.shape}) -> {base}')
    ann = {
        'video_id': base, 'label': gesture_name, 'variant': label,
        'frames_path': kp_path, 'roi_path': roi_path,
        'num_frames': len(frames), 'hand': hand,
    }
    return ann


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/my_annotations.json')
    parser.add_argument('--data_dir', default=OUTPUT_DIR_DEFAULT)
    parser.add_argument('--start_from', default='', help='Resume from gesture.variant (e.g. "One.index_front")')
    parser.add_argument('--hand', default='Right', choices=['Left', 'Right', 'Both'],
                        help='Which hand to record (default: Right). Both = bimanual')
    parser.add_argument('--simple', action='store_true',
                        help='Single condition mode (NORMAL/MID/CENTER only, 15 samples)')
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    tracker = HandTracker(min_detection_confidence=0.35, min_presence_confidence=0.35,
                          min_tracking_confidence=0.25)
    recognizer = RuleRecognizer()
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    annotations = []
    sample_idx = 0

    # Count totals
    total_variants = sum(len(v) for v in GESTURE_VARIANTS.values())
    is_bimanual = (args.hand == 'Both')

    if is_bimanual or args.simple:
        mode_name = 'Bimanual' if is_bimanual else f'Simple ({args.hand} hand)'
        total_samples = total_variants
        print(f'  Recorder v6.0 {mode_name} — {total_variants} variants x 1 condition = {total_samples} samples')
        print(f'  {args.hand} hand | {RECORD_SECONDS}s each | SPACE=record S=skip Q=quit')
        print()

        for gesture_name, variants in GESTURE_VARIANTS.items():
            for var_id, var_hint in variants:
                sample_idx += 1
                label = f'{gesture_name}-{var_id}_NORMAL_MID_CENTER_{sample_idx:04d}'
                prompt = [
                    (f'[{sample_idx}/{total_samples}] {gesture_name}/{var_id}  {args.hand} hand', (0, 255, 0)),
                    (f'{var_hint[:55]}  |  SPACE=rec  S=skip  Q=quit', (200, 200, 200)),
                ]
                action = wait_key(cap, prompt)
                if action == 'quit':
                    cap.release(); cv2.destroyAllWindows(); tracker.close()
                    _save(annotations, args.output); return
                if action == 'skip':
                    continue
                ann = record_sequence(cap, tracker, recognizer, args, label, gesture_name, args.hand)
                if isinstance(ann, dict):
                    annotations.append(ann)

        cap.release(); cv2.destroyAllWindows(); tracker.close()
        _save(annotations, args.output)
        return

    total_samples = total_variants * len(LIGHTS) * len(DISTS) * len(ANGLES)
    annotations = []
    sample_idx = 0
    skipped = False

    print(f'  Recorder v6.0 — {total_variants} variants x 3 lights x 3 dist x 3 angles = {total_samples} samples')
    print(f'  Record by LIGHT first: adjust light once, record all gestures, then switch')
    print(f'  {args.hand} hand | {RECORD_SECONDS}s per sample | SPACE=record S=skip Q=quit')
    print()

    for en_light, cn_light in LIGHTS:
        print(f'\n{"="*60}')
        print(f'  >>> LIGHT: {en_light} — adjust lighting now, then SPACE')
        print(f'{"="*60}')
        action = wait_key(cap, [
            (f'LIGHT = {en_light} — adjust room lighting now', (0, 255, 255)),
            ('SPACE=start this light  Q=quit', (0, 255, 0)),
        ])
        if action == 'quit':
            break

        for gesture_name, variants in GESTURE_VARIANTS.items():
            for var_id, var_hint in variants:
                # Resume support
                resume_key = f'{en_light}.{gesture_name}.{var_id}'
                if args.start_from and not skipped:
                    if resume_key != args.start_from:
                        continue
                    else:
                        skipped = True

                skip_gesture = False
                for dist in DISTS:
                    if skip_gesture:
                        break
                    for angle in ANGLES:
                        if skip_gesture:
                            break
                        sample_idx += 1
                        label = f'{gesture_name}-{var_id}_{cn_light}_{dist}_{angle}_{sample_idx:04d}'
                        short_hint = var_hint[:55]

                        prompt = [
                            (f'[{sample_idx}/{total_samples}] {gesture_name}/{var_id}  L={en_light}  D={dist}  A={angle}', (0, 255, 0)),
                            (f'{short_hint}  |  SPACE=rec  S=skip gesture  Q=quit', (200, 200, 200)),
                        ]
                        action = wait_key(cap, prompt)
                        if action == 'quit':
                            cap.release(); cv2.destroyAllWindows(); tracker.close()
                            _save(annotations, args.output); return
                        if action == 'skip':
                            skip_gesture = True
                            sample_idx -= 1
                            break

                        ann = record_sequence(cap, tracker, recognizer, args, label, gesture_name, args.hand)
                        if ann == 'retry':
                            sample_idx -= 1
                            continue
                        if isinstance(ann, dict):
                            annotations.append(ann)

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    _save(annotations, args.output)


def _save(annotations, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    print(f'\nDone! {len(annotations)} samples -> {output_path}')
    # Label distribution
    from collections import Counter
    labels = Counter(a['label'] for a in annotations)
    print('Label distribution:')
    for label, count in sorted(labels.items()):
        print(f'  {label}: {count}')


if __name__ == '__main__':
    main()
