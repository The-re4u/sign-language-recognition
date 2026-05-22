# coding:utf-8
"""
Gesture data recorder — SPACE to start, auto-stop 2s, no countdown.
Order: BRIGHT(all) -> NORMAL(all) -> DIM(all)
"""

import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from core.perception.hand_tracker import HandTracker, HandLandmarksWrapper
from core.fallback.rule_recognizer import RuleRecognizer

GESTURES = [
    'Closed_Fist', 'Good', 'Victory',
    'Eight', 'Two', 'Pinky_Up', 'One',
    'Three', 'Four', 'Open_Palm', 'Six', 'Seven', 'Nine',
]

LIGHTS = [
    ('BRIGHT', '强光'),
    ('NORMAL', '中光'),
    ('DIM', '弱光'),
]
ANGLES = ['LEFT', 'CENTER', 'RIGHT']
DISTS  = ['NEAR', 'MID', 'FAR']

TOTAL = len(LIGHTS) * len(GESTURES) * len(ANGLES) * len(DISTS)


def wait_space(cap):
    """Wait for SPACE keypress, return True; Q=quit returns False."""
    while True:
        ok, frame = cap.read()
        if not ok: return False
        frame = cv2.flip(frame, 1)
        cv2.imshow('Recorder', frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 32: return True   # SPACE
        if k == ord('q') or k == 27: return False  # Q / ESC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/my_annotations.json')
    parser.add_argument('--data_dir', default='data/my_sequences')
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    tracker = HandTracker(use_gpu=True)
    recognizer = RuleRecognizer()
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    annotations = []
    idx = 0

    print(f'SPACE=start  R=retry  S=skip  Q=quit  |  Total: {TOTAL}')

    for en_light, cn_light in LIGHTS:
        print(f'\n=== LIGHT: {en_light} ({cn_light}) — adjust now, then SPACE ===')
        if not wait_space(cap):
            cap.release(); cv2.destroyAllWindows(); tracker.close(); return

        for gesture in GESTURES:
            skip_gesture = False
            for angle in ANGLES:
                if skip_gesture: break
                for dist in DISTS:
                    if skip_gesture: break
                    idx += 1
                    label = f'{gesture}_{cn_light}_{angle}_{dist}'

                    # ---- Wait for SPACE ----
                    while True:
                        ok, frame = cap.read()
                        if not ok: break
                        frame = cv2.flip(frame, 1)
                        h, w = frame.shape[:2]
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = tracker.detect_video(rgb, int(time.time() * 1000))

                        # UI
                        cv2.rectangle(frame, (0, 0), (w, 115), (30, 30, 30), -1)
                        cv2.putText(frame, f'[{idx}/{TOTAL}] {gesture} | {en_light} | {angle} | {dist}',
                                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                        cv2.putText(frame, 'SPACE=record  R=retry  S=skip  Q=quit',
                                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                        if result.hand_landmarks:
                            w0 = HandLandmarksWrapper(result.hand_landmarks[0])
                            for lm in w0.landmark:
                                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 2, (0, 255, 0), -1)
                            _, _, hint = recognizer.recognize(w0)
                            c = (0, 255, 0) if hint == gesture else (0, 180, 255)
                            cv2.putText(frame, f'Hint: {hint}', (15, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)

                        cv2.imshow('Recorder', frame)
                        k = cv2.waitKey(30) & 0xFF

                        if k == 32:       # SPACE → record
                            break
                        elif k == ord('r'):  # retry
                            if annotations:
                                last = annotations.pop()
                                p = os.path.join(args.data_dir, last['frames_path'])
                                if os.path.exists(p): os.remove(p)
                                idx -= 1
                                print(f'  Retry: removed {last["label"]}')
                            # Also remove the one before if it was this same idx
                            idx -= 1
                            break  # redo this variation
                        elif k == ord('s'):  # skip gesture
                            skip_gesture = True
                            print(f'  Skip gesture: {gesture}')
                            idx -= 1
                            break
                        elif k == ord('q') or k == 27:
                            cap.release(); cv2.destroyAllWindows(); tracker.close()
                            with open(args.output, 'w', encoding='utf-8') as f:
                                json.dump(annotations, f, ensure_ascii=False, indent=2)
                            print(f'Quit. {len(annotations)} saved.')
                            return

                    if skip_gesture:
                        break

                    # ---- Record 2 seconds ----
                    frames = []
                    roi_frames = []
                    t0 = time.time()
                    while time.time() - t0 < 2.0:
                        ok, frame = cap.read()
                        if not ok: break
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = tracker.detect_video(rgb, int(time.time() * 1000))

                        r = max(0.0, 2.0 - (time.time() - t0))
                        cv2.putText(frame, f'REC {r:.1f}s', (frame.shape[1]//2-70, frame.shape[0]//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)

                        if result.hand_landmarks:
                            w0 = HandLandmarksWrapper(result.hand_landmarks[0])
                            kp = np.array([[lm.x, lm.y, lm.z] for lm in w0.landmark], dtype=np.float32)
                            frames.append(kp)
                            # Crop 96x96 hand ROI for visual modality
                            xs = [int(lm.x * frame.shape[1]) for lm in w0.landmark]
                            ys = [int(lm.y * frame.shape[0]) for lm in w0.landmark]
                            pad = 25
                            x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
                            x2, y2 = min(frame.shape[1], max(xs) + pad), min(frame.shape[0], max(ys) + pad)
                            roi = cv2.resize(frame[y1:y2, x1:x2], (96, 96))
                            roi_frames.append(roi)
                            for lm in w0.landmark:
                                cv2.circle(frame, (int(lm.x*frame.shape[1]), int(lm.y*frame.shape[0])),
                                           1, (0, 255, 0), -1)

                        cv2.imshow('Recorder', frame)
                        cv2.waitKey(1)

                    # Drain keys
                    for _ in range(15):
                        cv2.waitKey(1)

                    # Save
                    if len(frames) >= 6:
                        arr = np.array(frames, dtype=np.float32)
                        roi_arr = np.array(roi_frames, dtype=np.uint8)
                        name = f'{label}_{idx:04d}.npy'
                        roi_name = f'{label}_{idx:04d}_roi.npy'
                        np.save(os.path.join(args.data_dir, name), arr)
                        np.save(os.path.join(args.data_dir, roi_name), roi_arr)
                        annotations.append({
                            'video_id': name.replace('.npy', ''),
                            'label': gesture,
                            'variation': f'{cn_light}_{angle}_{dist}',
                            'frames_path': name,
                            'roi_path': roi_name,
                            'num_frames': len(frames),
                        })
                        print(f'  [{idx}/{TOTAL}] {label} ({len(frames)}f)')
                    else:
                        print(f'  [{idx}/{TOTAL}] {label} SKIP ({len(frames)}f)')
                        idx -= 1

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    print(f'\nDone! {len(annotations)} samples -> {args.output}')


if __name__ == '__main__':
    main()
