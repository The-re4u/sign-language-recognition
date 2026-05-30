# coding:utf-8
"""Live camera test for E3 model (SpatialGCN + HandShapeContext)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import torch

from core.perception.hand_tracker import HandTracker
from core.feature.spatial_gcn import SpatialGCN
from core.feature.hand_shape_context import HandShapeContext
from core.feature.multimodal_fusion import CrossModalFusion
from core.temporal.slowfast_tcn import SlowFastTCN

# ── Label mapping (must match training data) ──
IDX_TO_LABEL = {
    0: 'Closed_Fist', 1: 'Eight', 2: 'Four', 3: 'Nine', 4: 'One',
    5: 'Open_Palm', 6: 'Seven', 7: 'Six', 8: 'Three', 9: 'Two',
}

CHECKPOINT = 'models/checkpoints/E3_spatial_geo/best_model.pth'
WINDOW = 32          # frames in sliding window
INFER_EVERY = 4      # run inference every N frames (higher = faster)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load E3 model ──
spatial = SpatialGCN().to(device).eval()
geometric = HandShapeContext().to(device).eval()
fusion = CrossModalFusion(visual_dim=256, motion_dim=256).to(device).eval()
tcn = SlowFastTCN(input_dim=256, num_classes=10).to(device).eval()

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
spatial.load_state_dict(ckpt['spatial_model'])
geometric.load_state_dict(ckpt['geometric_model'])
fusion.load_state_dict(ckpt['fusion_model'])
tcn.load_state_dict(ckpt['tcn_model'])
print(f'Loaded: {CHECKPOINT}')

# ── Camera ──
tracker = HandTracker(min_detection_confidence=0.35, min_presence_confidence=0.35,
                      min_tracking_confidence=0.25)
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print('E3 Live Test — Q=quit')
print(f'Model: SpatialGCN + HandShapeContext (2.80M params)')
print(f'Window: {WINDOW} frames, inference every {INFER_EVERY} frames')

buffer = []           # list of [21,3] keypoint arrays
pred_text = '...'     # current prediction text
pred_conf = 0.0       # confidence (softmax max)
frame_count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = tracker.detect_video(rgb, int(time.time() * 1000))

    # ── Collect keypoints ──
    kp = None
    if result.hand_landmarks:
        # Use first detected hand
        lm = result.hand_landmarks[0]
        kp = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)  # [21, 3]
        buffer.append(kp)
        if len(buffer) > WINDOW:
            buffer.pop(0)

    # ── Inference ──
    if len(buffer) >= 16 and frame_count % INFER_EVERY == 0:
        seq = np.stack(buffer[-WINDOW:], axis=0)  # [T, 21, 3]
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)  # [1, T, 21, 3]
        B, T = seq_t.shape[:2]

        with torch.no_grad():
            kp_flat = seq_t.reshape(B * T, 21, 3)
            spa = spatial(kp_flat).reshape(B, T, 256)
            geo = geometric(seq_t)
            mot = torch.zeros(B, T, 256, device=device)

            feats = []
            for t_idx in range(T):
                f = fusion(geo[:, t_idx, :], spa[:, t_idx, :], mot[:, t_idx, :])
                feats.append(f.unsqueeze(1))
            fused_seq = torch.cat(feats, dim=1)

            logits = tcn(fused_seq).mean(dim=1)  # [1, 10]
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            pred_text = IDX_TO_LABEL.get(pred_idx, '?')
            pred_conf = probs[0, pred_idx].item()

    frame_count += 1

    # ── Draw ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        for m in lm:
            cv2.circle(frame, (int(m.x * w), int(m.y * h)), 3, (0, 255, 100), -1)

    # Prediction
    color = (0, 255, 0) if pred_conf > 0.6 else (0, 200, 255) if pred_conf > 0.3 else (0, 0, 255)
    cv2.putText(frame, f'{pred_text}  ({pred_conf:.0%})', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # Top-3
    if hasattr(torch, 'topk') and 'probs' in dir():
        try:
            top3 = torch.topk(probs[0], 3)
            for k in range(3):
                name = IDX_TO_LABEL.get(top3.indices[k].item(), '?')
                cv2.putText(frame, f'{name}: {top3.values[k].item():.0%}',
                            (10, 70 + k * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        except Exception:
            pass

    # Buffer status
    buf_pct = len(buffer) / WINDOW * 100
    cv2.putText(frame, f'Buffer: {len(buffer)}/{WINDOW} ({buf_pct:.0f}%)',
                (w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('E3 Live Test (SpatialGCN + HandShapeContext)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tracker.close()
print('Done.')
