# coding:utf-8
"""Live test v5 RuleRecognizer with camera — all CSL digits 0-9."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from core.perception.hand_tracker import HandTracker, HandLandmarksWrapper
from core.fallback.rule_recognizer import RuleRecognizer, each_finger_up

tracker = HandTracker(min_detection_confidence=0.35, min_presence_confidence=0.35,
                      min_tracking_confidence=0.25)
recognizer = RuleRecognizer()
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print('RuleRecognizer v5 — CSL 0-9 Live Test')
print('Q=quit')

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = tracker.detect_video(rgb, int(time.time() * 1000))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)

    if result.hand_landmarks:
        for i in range(len(result.hand_landmarks)):
            w0 = HandLandmarksWrapper(result.hand_landmarks[i])
            label = result.handedness[i][0].category_name
            fu = each_finger_up(w0)
            gesture = recognizer.recognize(w0)[2]

            clr = (255, 100, 0) if label == 'Left' else (0, 255, 100)
            for lm in w0.landmark:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, clr, -1)

            # Gesture result
            g_color = (0, 255, 0) if gesture != '?' else (0, 0, 255)
            cv2.putText(frame, f'{label}: {gesture}  (n={sum(fu)})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, g_color, 2)

            # Finger states
            fnames = ['T', 'I', 'M', 'R', 'P']
            y0 = 55
            for j, (name, up) in enumerate(zip(fnames, fu)):
                fx = 10 + j * 55
                clr_f = (0, 255, 0) if up else (100, 100, 100)
                cv2.putText(frame, f'{name}:{"UP" if up else "dn"}', (fx, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr_f, 1)

            # 0-9 status grid
            y0 = 75
            digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            for j, d in enumerate(digits):
                x = 10 + (j % 5) * 65
                yy = y0 + (j // 5) * 22
                match = (gesture == ('Zero' if d == '0' else
                          'One' if d == '1' else 'Two' if d == '2' else
                          'Three' if d == '3' else 'Four' if d == '4' else
                          'Five' if d == '5' else 'Six' if d == '6' else
                          'Seven' if d == '7' else 'Eight' if d == '8' else 'Nine'))
                c = (0, 255, 0) if match else (60, 60, 60)
                cv2.putText(frame, d, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    else:
        cv2.putText(frame, 'No hand detected', (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('CSL 0-9 v5 Rule Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tracker.close()
