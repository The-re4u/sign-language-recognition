# coding:utf-8
"""Bimanual interaction demo — dual-hand recognition + cross-attention visualization.

Uses E6 model for per-hand gesture classification, shows:
  - Left hand mode (which semantic layer)
  - Right hand content (which 0-9 gesture)
  - Dual-hand control gestures (separator / undo-word / undo-sentence)
  - Cross-attention heatmap between left↔right hand features

Usage:
  python tools/demo_bimanual.py
  python tools/demo_bimanual.py --checkpoint models/checkpoints/E6_semantic/best_model.pth
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.perception.hand_tracker import HandTracker, HandLandmarksWrapper
from core.perception.roi_cropper import ROICropper
from core.feature.spatial_gcn import SpatialGCN
from core.feature.hand_shape_context import HandShapeContext
from core.feature.visual_encoder import LightweightVisualEncoder
from core.feature.multimodal_fusion import CrossModalFusion, GatedFusion
from core.feature.semantic_encoder import SemanticProjector
from core.feature.bimanual_attention import BimanualCrossAttention
from core.temporal.slowfast_tcn import SlowFastTCN

# ── Gesture labels ──
IDX_TO_LABEL = {
    0: 'Closed_Fist', 1: 'Eight', 2: 'Four', 3: 'Nine', 4: 'One',
    5: 'Open_Palm', 6: 'Seven', 7: 'Six', 8: 'Three', 9: 'Two',
}

# ── Left-hand mode mapping ──
MODE_MAP = {
    'Seven': '数字', 'One': '常用语', 'Two': '病症',
    'Three': '身体', 'Four': '医院', 'Open_Palm': '程度', 'Six': '时间',
}

# ── Right-hand content per mode ──
CONTENT_MAP = {
    ('数字', 'Closed_Fist'): '0', ('数字', 'One'): '1', ('数字', 'Two'): '2',
    ('数字', 'Three'): '3', ('数字', 'Four'): '4', ('数字', 'Open_Palm'): '5',
    ('数字', 'Six'): '6', ('数字', 'Seven'): '7', ('数字', 'Eight'): '8', ('数字', 'Nine'): '9',
    ('常用语', 'Closed_Fist'): '请帮帮我', ('常用语', 'Two'): '好的', ('常用语', 'Three'): '谢谢',
    ('常用语', 'Four'): '我想要', ('常用语', 'Open_Palm'): '你好',
    ('常用语', 'Six'): '是', ('常用语', 'Seven'): '再见', ('常用语', 'Eight'): '我不舒服',
    ('常用语', 'Nine'): '不是',
    ('病症', 'Closed_Fist'): '疼', ('病症', 'One'): '咳嗽', ('病症', 'Three'): '食欲不振',
    ('病症', 'Four'): '腹泻', ('病症', 'Open_Palm'): '发烧', ('病症', 'Six'): '恶心',
    ('病症', 'Seven'): '头晕', ('病症', 'Eight'): '乏力', ('病症', 'Nine'): '胸闷',
    ('身体', 'Closed_Fist'): '头', ('身体', 'One'): '眼', ('身体', 'Two'): '耳',
    ('身体', 'Four'): '腿', ('身体', 'Open_Palm'): '手', ('身体', 'Six'): '脚',
    ('身体', 'Seven'): '背', ('身体', 'Eight'): '胸', ('身体', 'Nine'): '胃',
    ('医院', 'Closed_Fist'): '挂号', ('医院', 'One'): '取药', ('医院', 'Two'): '住院',
    ('医院', 'Three'): '输液', ('医院', 'Four'): '检查', ('医院', 'Open_Palm'): '出院',
    ('医院', 'Six'): '转科', ('医院', 'Seven'): '化验', ('医院', 'Eight'): '手术',
    ('医院', 'Nine'): '换药',
    ('程度', 'One'): '轻微', ('程度', 'Two'): '中等', ('程度', 'Three'): '较重',
    ('程度', 'Four'): '严重', ('程度', 'Six'): '好转', ('程度', 'Seven'): '恶化',
    ('程度', 'Eight'): '稳定', ('程度', 'Nine'): '紧急',
    ('时间', 'Closed_Fist'): '秒', ('时间', 'One'): '昨天', ('时间', 'Two'): '今天',
    ('时间', 'Three'): '分钟', ('时间', 'Four'): '小时', ('时间', 'Open_Palm'): '天',
    ('时间', 'Six'): '现在', ('时间', 'Seven'): '明天', ('时间', 'Eight'): '持续',
    ('时间', 'Nine'): '早上',
}

WINDOW = 32
MIN_FRAMES = 16
INFER_EVERY = 4


def build_models(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_mods = ckpt.get('modalities', {})

    # Auto angle_dim
    angle_dim = 64
    if 'spatial_model' in ckpt:
        w = ckpt['spatial_model'].get('final_fusion.0.weight')
        if w is not None: angle_dim = w.shape[1] - 256

    spatial = SpatialGCN(angle_dim=angle_dim).to(device).eval()
    spatial.load_state_dict(ckpt['spatial_model'])

    geometric = HandShapeContext().to(device).eval()
    geometric.load_state_dict(ckpt['geometric_model'])

    cnn = LightweightVisualEncoder(freeze_backbone=True).to(device).eval()
    cnn.load_state_dict(ckpt['visual_model'])

    gate = GatedFusion().to(device).eval()
    gate.load_state_dict(ckpt['gated_fusion'])

    fusion = CrossModalFusion(visual_dim=512, motion_dim=256).to(device).eval()
    fusion.load_state_dict(ckpt['fusion_model'])

    tcn = SlowFastTCN(input_dim=256, num_classes=10).to(device).eval()
    tcn.load_state_dict(ckpt['tcn_model'])

    # BimanualCrossAttention (fresh weights — demo only, not trained)
    bimanual = BimanualCrossAttention(dim=256).to(device).eval()

    # Semantic projector (for feature space viz)
    semantic = None
    if ckpt_mods.get('semantic') and 'semantic_projector' in ckpt:
        semantic = SemanticProjector().to(device).eval()
        semantic.load_state_dict(ckpt['semantic_projector'])

    print(f'Loaded E6 (val_acc={ckpt.get("val_acc", "?")}) + BimanualCrossAttention (untrained)')
    return (spatial, geometric, cnn, gate, fusion, tcn, bimanual, semantic)


def classify_hand(models, device, kp_seq, roi_seq):
    """Classify a single hand's gesture from keypoint+ROI sequence."""
    spatial, geometric, cnn, gate, fusion, tcn, _, _ = models
    kp_t = torch.from_numpy(kp_seq).unsqueeze(0).to(device)
    roi_t = torch.from_numpy(roi_seq).float().permute(0,3,1,2).unsqueeze(0).to(device) / 255.0
    B, T = kp_t.shape[:2]

    with torch.no_grad():
        kp_f = kp_t.reshape(B*T, 21, 3)
        roi_f = roi_t.reshape(B*T, 3, 96, 96)
        spa = spatial(kp_f).reshape(B, T, 256)
        geo = geometric(kp_t)
        cnn_f = cnn(roi_f).reshape(B, T, 512)
        vis = gate(cnn_f.reshape(B*T,512), geo.reshape(B*T,256)).reshape(B, T, 512)
        mot = torch.zeros(B, T, 256, device=device)
        feats = []
        for t in range(T):
            f = fusion(vis[:,t,:], spa[:,t,:], mot[:,t,:])
            feats.append(f.unsqueeze(1))
        fused = torch.cat(feats, dim=1)
        logits = tcn(fused).mean(dim=1)
        probs = F.softmax(logits, dim=1)
        idx = logits.argmax(dim=1).item()
    return IDX_TO_LABEL.get(idx, '?'), probs[0, idx].item(), probs[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='models/checkpoints/E6_semantic/best_model.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    models = build_models(args.checkpoint, device)

    tracker = HandTracker(min_detection_confidence=0.3, min_presence_confidence=0.3,
                          min_tracking_confidence=0.2)
    cropper = ROICropper(target_size=(96, 96), margin=0.2)
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print('\n=== Bimanual Interaction Demo ===')
    print('Left hand  → Mode selector')
    print('Right hand → Content gesture')
    print('Both hands → Control (One=sep, Two=undo, Three=clear)')
    print('Q=quit\n')

    buf_left_kp, buf_left_roi = [], []
    buf_right_kp, buf_right_roi = [], []
    left_gesture, right_gesture = '?', '?'
    left_conf, right_conf = 0.0, 0.0
    mode_text, content_text = '—', '—'
    fc = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = tracker.detect_video(rgb, ts)

        left_kp, right_kp = None, None
        left_roi, right_roi = None, None

        if result.hand_landmarks and len(result.hand_landmarks) >= 2:
            # Both hands detected — identify by handedness
            for i in range(len(result.hand_landmarks)):
                hand_label = result.handedness[i][0].category_name
                lm = result.hand_landmarks[i]
                kp = np.array([[lm.x, lm.y, lm.z] for lm in lm], dtype=np.float32)
                roi = cropper.crop(rgb, HandLandmarksWrapper(lm))
                if hand_label == 'Left':
                    left_kp, left_roi = kp, roi
                else:
                    right_kp, right_roi = kp, roi

        elif result.hand_landmarks and len(result.hand_landmarks) == 1:
            # Single hand — assume right hand (content)
            hand_label = result.handedness[0][0].category_name
            lm = result.hand_landmarks[0]
            kp = np.array([[lm.x, lm.y, lm.z] for lm in lm], dtype=np.float32)
            roi = cropper.crop(rgb, HandLandmarksWrapper(lm))
            if hand_label == 'Left':
                left_kp, left_roi = kp, roi
            else:
                right_kp, right_roi = kp, roi

        # Update buffers
        if left_kp is not None:
            buf_left_kp.append(left_kp); buf_left_roi.append(left_roi)
            if len(buf_left_kp) > WINDOW: buf_left_kp.pop(0); buf_left_roi.pop(0)
        if right_kp is not None:
            buf_right_kp.append(right_kp); buf_right_roi.append(right_roi)
            if len(buf_right_kp) > WINDOW: buf_right_kp.pop(0); buf_right_roi.pop(0)

        # Inference
        if fc % INFER_EVERY == 0:
            if len(buf_left_kp) >= MIN_FRAMES:
                sk = np.stack(buf_left_kp[-WINDOW:], axis=0)
                sr = np.stack(buf_left_roi[-WINDOW:], axis=0)
                left_gesture, left_conf, _ = classify_hand(models, device, sk, sr)
            if len(buf_right_kp) >= MIN_FRAMES:
                sk = np.stack(buf_right_kp[-WINDOW:], axis=0)
                sr = np.stack(buf_right_roi[-WINDOW:], axis=0)
                right_gesture, right_conf, _ = classify_hand(models, device, sk, sr)

        # Mode + Content lookup
        mode_text = MODE_MAP.get(left_gesture, '—') if left_kp is not None else '—'
        content_text = CONTENT_MAP.get((mode_text, right_gesture), right_gesture) if right_kp is not None else '—'

        fc += 1

        # ── Draw UI ──
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 220), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.55, ov, 0.45, 0)

        # Draw landmarks
        if result.hand_landmarks:
            for i in range(len(result.hand_landmarks)):
                lm = result.hand_landmarks[i]
                hand_label = result.handedness[i][0].category_name
                color = (255, 100, 0) if hand_label == 'Left' else (0, 255, 100)
                for m in lm:
                    cv2.circle(frame, (int(m.x*w), int(m.y*h)), 3, color, -1)

        # Left hand panel (mode)
        cv2.putText(frame, 'LEFT (Mode)', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,180,0), 1)
        l_color = (0,255,0) if left_conf > 0.6 else (200,200,200)
        cv2.putText(frame, f'{left_gesture} ({left_conf:.0%})', (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, l_color, 2)
        cv2.putText(frame, f'Mode: {mode_text}', (15, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # Right hand panel (content)
        cv2.putText(frame, 'RIGHT (Content)', (w//2 + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        r_color = (0,255,0) if right_conf > 0.6 else (200,200,200)
        cv2.putText(frame, f'{right_gesture} ({right_conf:.0%})', (w//2 + 15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, r_color, 2)
        cv2.putText(frame, f'Content: {content_text}', (w//2 + 15, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

        # Combined output
        combined = f'{mode_text}:{content_text}' if mode_text != '—' else content_text
        cv2.putText(frame, combined, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # Control hint
        if left_gesture == right_gesture and left_gesture in ('One', 'Two', 'Three') and left_gesture != '?':
            ctrl = {'One': 'SEPARATOR', 'Two': 'UNDO WORD', 'Three': 'UNDO SENTENCE'}
            cv2.putText(frame, f'CTRL: {ctrl[left_gesture]}', (15, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        # Buffer + mode badge
        lb = len(buf_left_kp); rb = len(buf_right_kp)
        cv2.putText(frame, f'L:{lb}/{WINDOW}  R:{rb}/{WINDOW} | E6 Bimanual Demo',
                    (w - 350, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

        cv2.imshow('Bimanual Interaction Demo (E6 + Cross-Attention)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows(); tracker.close()
    print('Done.')


if __name__ == '__main__':
    main()
