# coding:utf-8
"""E4 live test — SpatialGCN + CNN + HandShapeContext + GatedFusion.

Usage:
  python tools/test_e4_live.py                 # interactive mode prompt
  python tools/test_e4_live.py --image a.jpg   # single image inference
  python tools/test_e4_live.py --camera         # camera mode directly
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import torch
import argparse

from core.perception.hand_tracker import HandTracker, HandLandmarksWrapper
from core.perception.roi_cropper import ROICropper
from core.feature.spatial_gcn import SpatialGCN
from core.feature.hand_shape_context import HandShapeContext
from core.feature.visual_encoder import LightweightVisualEncoder
from core.feature.multimodal_fusion import CrossModalFusion, GatedFusion
from core.temporal.slowfast_tcn import SlowFastTCN

IDX_TO_LABEL = {
    0: 'Closed_Fist', 1: 'Eight', 2: 'Four', 3: 'Nine', 4: 'One',
    5: 'Open_Palm', 6: 'Seven', 7: 'Six', 8: 'Three', 9: 'Two',
}
CHECKPOINT = 'models/checkpoints/E4_spatial_cnn_geo/best_model.pth'
WINDOW = 32
MIN_FRAMES = 16
INFER_EVERY = 4


def build_and_load(ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    spatial = SpatialGCN().to(device).eval()
    geometric = HandShapeContext().to(device).eval()
    cnn = LightweightVisualEncoder(freeze_backbone=True).to(device).eval()
    gate = GatedFusion().to(device).eval()
    fusion = CrossModalFusion(visual_dim=512, motion_dim=256).to(device).eval()
    tcn = SlowFastTCN(input_dim=256, num_classes=10).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    spatial.load_state_dict(ckpt['spatial_model'])
    geometric.load_state_dict(ckpt['geometric_model'])
    cnn.load_state_dict(ckpt['visual_model'])
    gate.load_state_dict(ckpt['gated_fusion'])
    fusion.load_state_dict(ckpt['fusion_model'])
    tcn.load_state_dict(ckpt['tcn_model'])
    print(f'Loaded E4 checkpoint (val_acc={ckpt.get("val_acc", "?")})\n')
    return (spatial, geometric, cnn, gate, fusion, tcn), device


def infer(models, device, seq_kp, seq_roi):
    """seq_kp: [T,21,3] np, seq_roi: [T,96,96,3] uint8 → (label, conf, top3)"""
    spatial, geometric, cnn, gate, fusion, tcn = models

    kp_t = torch.from_numpy(seq_kp).unsqueeze(0).to(device)           # [1,T,21,3]
    roi_t = torch.from_numpy(seq_roi).float().permute(0,3,1,2).unsqueeze(0).to(device) / 255.0  # [1,T,3,96,96]
    B, T = kp_t.shape[:2]

    with torch.no_grad():
        kp_f = kp_t.reshape(B * T, 21, 3)
        roi_f = roi_t.reshape(B * T, 3, 96, 96)

        spa = spatial(kp_f).reshape(B, T, 256)
        geo = geometric(kp_t)
        cnn_f = cnn(roi_f).reshape(B, T, 512)

        vis = gate(cnn_f.reshape(B*T, 512), geo.reshape(B*T, 256)).reshape(B, T, 512)
        mot = torch.zeros(B, T, 256, device=device)

        feats = []
        for t in range(T):
            f = fusion(vis[:, t, :], spa[:, t, :], mot[:, t, :])
            feats.append(f.unsqueeze(1))
        fused = torch.cat(feats, dim=1)
        logits = tcn(fused).mean(dim=1)
        probs = torch.softmax(logits, dim=1)
        idx = logits.argmax(dim=1).item()
        top3 = torch.topk(probs[0], 3)

    label = IDX_TO_LABEL.get(idx, '?')
    conf = probs[0, idx].item()
    top3_list = [(IDX_TO_LABEL.get(top3.indices[k].item(), '?'), top3.values[k].item()) for k in range(3)]
    return label, conf, top3_list


def image_mode(models, device, image_path):
    """Single-image inference."""
    tracker = HandTracker(min_detection_confidence=0.3, min_presence_confidence=0.3, min_tracking_confidence=0.2)
    cropper = ROICropper(target_size=(96, 96), margin=0.2)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f'ERROR: Cannot read: {image_path}')
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = tracker.detect_image(rgb)

    if not result.hand_landmarks:
        print('No hand detected.')
        tracker.close()
        return

    lm = result.hand_landmarks[0]
    kp = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
    wrapper = HandLandmarksWrapper(lm)
    roi = cropper.crop(rgb, wrapper)

    # Replicate to MIN_FRAMES for TCN
    seq_kp = np.tile(kp[None], (MIN_FRAMES, 1, 1))
    seq_roi = np.tile(roi[None], (MIN_FRAMES, 1, 1, 1))

    label, conf, top3 = infer(models, device, seq_kp, seq_roi)
    print(f'Result: {label} ({conf:.1%})')
    for name, p in top3:
        print(f'  {name:15s} {p:.1%}')
    tracker.close()


def camera_mode(models, device):
    """Live camera inference."""
    tracker = HandTracker(min_detection_confidence=0.3, min_presence_confidence=0.3, min_tracking_confidence=0.2)
    cropper = ROICropper(target_size=(96, 96), margin=0.2)

    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print('E4 Camera | Q=quit | Model: GCN+CNN+Geo+GatedFusion (6.89M)\n')

    buf_kp, buf_roi = [], []
    pred_text, pred_conf, top3_list = '...', 0.0, []
    fc = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = tracker.detect_video(rgb, ts)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            kp = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
            roi = cropper.crop(rgb, HandLandmarksWrapper(lm))
            buf_kp.append(kp)
            buf_roi.append(roi)
            if len(buf_kp) > WINDOW:
                buf_kp.pop(0); buf_roi.pop(0)

        if len(buf_kp) >= MIN_FRAMES and fc % INFER_EVERY == 0:
            sk = np.stack(buf_kp[-WINDOW:], axis=0)
            sr = np.stack(buf_roi[-WINDOW:], axis=0)
            pred_text, pred_conf, top3_list = infer(models, device, sk, sr)

        fc += 1

        # Draw
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.5, ov, 0.5, 0)

        if result.hand_landmarks:
            for lm in result.hand_landmarks[0]:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0,255,100), -1)

        clr = (0,255,0) if pred_conf>0.6 else (0,200,255) if pred_conf>0.3 else (0,0,255)
        cv2.putText(frame, f'{pred_text} ({pred_conf:.0%})', (15,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, clr, 2)
        for k, (nm, pb) in enumerate(top3_list[:3]):
            cv2.putText(frame, f'{nm}: {pb:.0%}', (15,75+k*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        buf_pct = len(buf_kp)/WINDOW*100
        cv2.putText(frame, f'E4 | Buff:{len(buf_kp)}/{WINDOW}', (w-220,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow('E4 GatedFusion v2', frame)
        if cv2.waitKey(1)&0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows(); tracker.close()
    print('Done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, default=None, help='Image path for single-image mode')
    p.add_argument('--camera', action='store_true', help='Camera mode directly')
    args = p.parse_args()

    models, device = build_and_load(CHECKPOINT)

    if args.image:
        image_mode(models, device, args.image)
    elif args.camera:
        camera_mode(models, device)
    else:
        print('E4 Test — choose mode:')
        print('  1. Camera (live)')
        print('  2. Image  (upload)')
        choice = input('Enter 1 or 2: ').strip()
        if choice == '2':
            path = input('Image path: ').strip()
            image_mode(models, device, path)
        else:
            camera_mode(models, device)
