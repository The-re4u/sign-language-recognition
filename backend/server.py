# coding:utf-8
"""FastAPI WebSocket backend for hospital sign language input method v3.0.

Architecture:
  Vue frontend (camera + UI) <--WebSocket--> FastAPI backend (recognition pipeline)

Receives frames as JPEG bytes, returns recognition results as JSON.
Stateless per-frame design — SentenceRecorder state is maintained on the server.
"""
import sys, os, json, time, base64, io, asyncio, threading
import numpy as np
import cv2

def _py(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.bool_,)): return bool(val)
    if isinstance(val, (np.integer,)): return int(val)
    if isinstance(val, (np.floating,)): return float(val)
    if isinstance(val, np.ndarray): return val.tolist()
    if isinstance(val, (list, tuple)): return [_py(v) for v in val]
    if isinstance(val, dict): return {k: _py(v) for k, v in val.items()}
    return val

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure working directory is project root (for model files, config, etc.)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn


class SetApiKeyRequest(BaseModel):
    api_key: str

# ---- Init recognition pipeline ----
from core.perception.hand_tracker import HandTracker, HandLandmarksWrapper
from core.perception.flow_estimator import FlowEstimator
from core.fallback.rule_recognizer import RuleRecognizer
from core.semantic.temporal_parser import TemporalSemanticParser
from core.semantic.sentence_recorder import SentenceRecorder
from core.semantic.deepseek_client import DeepSeekClient
from core.feature.kinematic_features import compute_all_kinematic_features

# ---- Init DL pipeline (multi-model support: E3/E4/E6) ----
_dl_available = False
_dl_buffer_left = None
_dl_buffer_right = None
_dl_hand_results = {}
_dl_buffer_lock = None
_dl_gesture_names = {}
_current_model = 'e6'  # 'e3' | 'e4' | 'e6' | 'rule'

# E3 components (GCN + Geo, no CNN, 256-dim visual slot)
_e3_spatial = None
_e3_geometric = None
_e3_fusion = None
_e3_onnx = None

# E4/E6 components (GCN + Geo + CNN + GatedFusion, 512-dim visual slot)
# Each model loads its own extractors — E6 weights diverge from E4 due to contrastive loss
_e4_spatial = None
_e4_geometric = None
_e4_visual = None
_e4_gated = None
_e4_fusion = None
_e4_onnx = None
_e6_spatial = None
_e6_geometric = None
_e6_visual = None
_e6_gated = None
_e6_fusion = None
_e6_onnx = None

try:
    import torch
    from collections import deque
    from core.feature.spatial_gcn import SpatialGCN
    from core.feature.hand_shape_context import HandShapeContext
    from core.feature.multimodal_fusion import CrossModalFusion, GatedFusion
    from core.feature.visual_encoder import LightweightVisualEncoder
    from core.inference.onnx_recognizer import ONNXRecognizer

    # ── Load E3 checkpoint ──
    _e3_ckpt_path = 'models/checkpoints/E3_spatial_geo/best_model.pth'
    _e3_ckpt = torch.load(_e3_ckpt_path, map_location='cpu', weights_only=True)
    _e3_angle_dim = 64
    if 'spatial_model' in _e3_ckpt:
        w = _e3_ckpt['spatial_model'].get('final_fusion.0.weight')
        if w is not None: _e3_angle_dim = w.shape[1] - 256
    _e3_spatial = SpatialGCN(angle_dim=_e3_angle_dim).eval()
    _e3_spatial.load_state_dict(_e3_ckpt['spatial_model'])
    _e3_geometric = HandShapeContext().eval()
    _e3_geometric.load_state_dict(_e3_ckpt['geometric_model'])
    _e3_fusion = CrossModalFusion(visual_dim=256, motion_dim=256).eval()
    _e3_fusion.load_state_dict(_e3_ckpt['fusion_model'])
    _e3_onnx = ONNXRecognizer('models/sign_recognizer_e3.onnx', num_classes=10)
    print(f'[Backend] E3 loaded (GCN+Geo, {_e3_angle_dim}-dim angles)')

    # ── Load E4 checkpoint ──
    _e4_ckpt_path = 'models/checkpoints/E4_spatial_cnn_geo/best_model.pth'
    _e4_ckpt = torch.load(_e4_ckpt_path, map_location='cpu', weights_only=True)
    _e4_angle_dim = 64
    if 'spatial_model' in _e4_ckpt:
        w = _e4_ckpt['spatial_model'].get('final_fusion.0.weight')
        if w is not None: _e4_angle_dim = w.shape[1] - 256
    _e4_spatial = SpatialGCN(angle_dim=_e4_angle_dim).eval()
    _e4_spatial.load_state_dict(_e4_ckpt['spatial_model'])
    _e4_geometric = HandShapeContext().eval()
    _e4_geometric.load_state_dict(_e4_ckpt['geometric_model'])
    _e4_visual = LightweightVisualEncoder(freeze_backbone=True).eval()
    _e4_visual.load_state_dict(_e4_ckpt['visual_model'])
    _e4_gated = GatedFusion().eval()
    _e4_gated.load_state_dict(_e4_ckpt['gated_fusion'])
    _e4_fusion = CrossModalFusion(visual_dim=512, motion_dim=256).eval()
    _e4_fusion.load_state_dict(_e4_ckpt['fusion_model'])
    _e4_onnx = ONNXRecognizer('models/sign_recognizer_e4.onnx', num_classes=10)
    print(f'[Backend] E4 loaded (GCN+Geo+CNN+GatedFusion, {_e4_angle_dim}-dim angles)')

    # ── Load E6 checkpoint (own extractors: fine-tuned with contrastive loss) ──
    _e6_ckpt_path = 'models/checkpoints/E6_semantic/best_model.pth'
    _e6_ckpt = torch.load(_e6_ckpt_path, map_location='cpu', weights_only=True)
    _e6_angle_dim = 64
    if 'spatial_model' in _e6_ckpt:
        w = _e6_ckpt['spatial_model'].get('final_fusion.0.weight')
        if w is not None: _e6_angle_dim = w.shape[1] - 256
    _e6_spatial = SpatialGCN(angle_dim=_e6_angle_dim).eval()
    _e6_spatial.load_state_dict(_e6_ckpt['spatial_model'])
    _e6_geometric = HandShapeContext().eval()
    _e6_geometric.load_state_dict(_e6_ckpt['geometric_model'])
    _e6_visual = LightweightVisualEncoder(freeze_backbone=True).eval()
    _e6_visual.load_state_dict(_e6_ckpt['visual_model'])
    _e6_gated = GatedFusion().eval()
    _e6_gated.load_state_dict(_e6_ckpt['gated_fusion'])
    _e6_fusion = CrossModalFusion(visual_dim=512, motion_dim=256).eval()
    _e6_fusion.load_state_dict(_e6_ckpt['fusion_model'])
    _e6_onnx = ONNXRecognizer('models/sign_recognizer_e6.onnx', num_classes=10)
    print(f'[Backend] E6 loaded (GCN+Geo+CNN+GatedFusion+Semantic, {_e6_angle_dim}-dim angles)')

    _dl_buffer_left = deque(maxlen=32)
    _dl_buffer_right = deque(maxlen=32)
    _dl_buffer_lock = threading.Lock()
    _dl_available = True
    print('[Backend] DL pipeline loaded: multi-model (E3/E4/E6)')
except ImportError as e:
    print(f'[Backend] DL pipeline not available (missing torch/onnx): {e}')
except Exception as e:
    print(f'[Backend] DL pipeline init failed: {e}')
    import traceback as _tb_init
    _tb_init.print_exc()

# Load DL gesture name mapping from training data
try:
    import json as _json3
    with open('data/train.json', 'r', encoding='utf-8') as _f:
        _train_data = _json3.load(_f)
    _labels = sorted(set(a.get('label', a.get('gesture', 'unknown')) for a in _train_data))
    _dl_gesture_names = {i: name for i, name in enumerate(_labels)}
    print(f'[Backend] DL gesture names: {_dl_gesture_names}')
except Exception as _e:
    print(f'[Backend] Failed to load gesture names: {_e}')

print('[Backend] Loading models...')
hand_tracker = HandTracker(min_detection_confidence=0.35, min_presence_confidence=0.35,
                           min_tracking_confidence=0.25)
flow_estimator = FlowEstimator()
rule_recognizer = RuleRecognizer()
semantic_parser = TemporalSemanticParser('config/hospital_chains.json')
semantic_parser.timeout = 4.0  # v3.1: was 1.5, too aggressive — auto-flushed when hands briefly lost
deepseek_client = DeepSeekClient()
sentence_recorder = SentenceRecorder(
    semantic_parser, cooldown=0.22, timeout=15.0,
    deepseek_client=deepseek_client, min_gesture_duration=0.2
)

# State
_frame_counter = 0
_input_skip = 2            # process every 2nd frame (25→12.5 FPS internal, 25 FPS output)
_flow_skip = 1               # optical flow enabled — used for real Farneback flow in DL
_cached_flow = None
_fps_estimate = 30.0
_last_frame_time = 0
_prev_keypoints = {}
_kp_history = {}
# Performance tracking (per session, reset on each start)
_perf_fps_sum = 0.0
_perf_detect_sum = 0.0
_perf_proc_sum = 0.0
_perf_frame_count = 0
_perf_session_start = 0.0
_perf_snapshot = False
_perf_dl_conf_sum = 0.0
_perf_dl_conf_count = 0
_kinematic_history_len = 16
_kin_skip = 6              # kinematic features every 6 processed frames
_sr_skip = 1               # sentence recorder on every processed frame (input_skip already gates)
_last_sr_result = None
_last_proc_result = None   # cached full proc_result for skipped frames
_last_hand_count = 0       # debounce flicker: track consistent hand count
_hand_debounce_timer = 0
_work_mode = 'translate'   # 'translate' = sign language translation, 'triage' = AI triage
_swap_hands = False        # v3.4: False=left mode right content, True=swapped
_triage_history = []       # conversation history for triage mode
_video_timestamp_ms = 0    # v3.1: increasing timestamp for VIDEO mode tracking
_first_frame_after_reconnect = True  # v3.2: IMAGE mode on first frame after WS reconnect

print('[Backend] Pipeline ready.')


def _dl_process():
    """Background daemon: PyTorch feature extraction + ONNX inference.

    Supports E3 (GCN+Geo), E4 (GCN+CNN+Geo+GatedFusion), E6 (E4+Semantic TCN).
    Switches feature extractors and ONNX model based on _current_model global.
    """
    import time as _t, traceback as _tb
    while True:
        _t.sleep(0.5)
        if _current_model == 'rule' or not _dl_available:
            continue
        try:
            # ── Select active components ──
            is_e3 = (_current_model == 'e3')
            is_e6 = (_current_model == 'e6')
            if is_e3:
                spatial, geometric = _e3_spatial, _e3_geometric
                visual_enc, gated = None, None
                fusion, onnx = _e3_fusion, _e3_onnx
            elif is_e6:
                spatial, geometric = _e6_spatial, _e6_geometric
                visual_enc, gated = _e6_visual, _e6_gated
                fusion, onnx = _e6_fusion, _e6_onnx
            else:
                spatial, geometric = _e4_spatial, _e4_geometric
                visual_enc, gated = _e4_visual, _e4_gated
                fusion, onnx = _e4_fusion, _e4_onnx

            for side, buf in [('Left', _dl_buffer_left), ('Right', _dl_buffer_right)]:
                if _dl_buffer_lock is None:
                    continue
                with _dl_buffer_lock:
                    if len(buf) < 1:
                        continue
                    items = list(buf)[-16:]
                    buf.clear()
                if len(items) < 16:
                    pad_n = 16 - len(items)
                    items = items + [items[-1]] * pad_n
                kps = np.stack([it['curr_kp'] for it in items])  # [T, 21, 3]
                with torch.no_grad():
                    kp_t = torch.from_numpy(kps).float()
                    T = kp_t.shape[0]

                    # ── Skeleton: SpatialGCN ──
                    spa_seq = spatial(kp_t)  # [T, 256]

                    # ── Geometric: HandShapeContext ──
                    geo_seq = geometric(kp_t)  # [T, 256]

                    # ── Visual: MobileNetV3 CNN (E4/E6 only) ──
                    vis_slot = None
                    if not is_e3:
                        roi_list = [it.get('roi_bgr') for it in items]
                        roi_stack = []
                        for roi_bgr in roi_list:
                            if roi_bgr is not None:
                                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                                roi_t = torch.from_numpy(roi_rgb).float().permute(2, 0, 1) / 255.0
                                roi_stack.append(roi_t)
                            else:
                                roi_stack.append(torch.zeros(3, 96, 96))
                        roi_tensor = torch.stack(roi_stack)  # [T, 3, 96, 96]
                        cnn_seq = visual_enc(roi_tensor)     # [T, 512]
                        vis_slot = gated(cnn_seq, geo_seq)    # [T, 512]
                    else:
                        vis_slot = geo_seq  # E3: geometric IS the visual slot

                    # ── Motion: zero (static gesture) ──
                    motion_seq = torch.zeros(T, 256)

                    # ── CrossModalFusion ──
                    fused_seq = []
                    for t in range(T):
                        f = fusion(vis_slot[t:t+1], spa_seq[t:t+1], motion_seq[t:t+1])
                        fused_seq.append(f.squeeze(0).cpu().numpy().astype(np.float32))

                if len(fused_seq) >= 1:
                    feat_array = np.stack(fused_seq)
                    result = onnx.predict_sequence_label(feat_array)
                    if result is not None:
                        lid, conf = result
                        name = _dl_gesture_names.get(lid, str(lid))
                        with _dl_buffer_lock:
                            _dl_hand_results[side] = (name, round(float(conf) * 100), time.time())
                        global _perf_dl_conf_sum, _perf_dl_conf_count
                        _perf_dl_conf_sum += float(conf)
                        _perf_dl_conf_count += 1
                    else:
                        name = '?'; conf = 0.0
                    print(f'[DL:{_current_model.upper()}] {side}: {name} conf={float(conf):.2f}')
        except Exception:
            print('[DL] Background thread error:')
            _tb.print_exc()

threading.Thread(target=_dl_process, daemon=True).start()

app = FastAPI(title='Hand Sign Language API v3.1')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
app.mount('/assets', StaticFiles(directory='assets'), name='assets')


def _landmarks_to_array(wrapper):
    return np.array([[lm.x, lm.y, lm.z] for lm in wrapper.landmark], dtype=np.float32)


def _process_frame(image_bgr):
    """Full frame processing pipeline — v3.1 low-latency edition."""
    global _frame_counter, _cached_flow, _fps_estimate, _last_frame_time, _perf_fps_sum, _perf_detect_sum, _perf_proc_sum, _perf_frame_count, _perf_session_start, _perf_snapshot
    global _prev_keypoints, _kp_history, _last_sr_result, _last_proc_result
    global _video_timestamp_ms, _first_frame_after_reconnect
    global _dl_buffer_left, _dl_buffer_right, _dl_hand_results, _last_hand_count, _hand_debounce_timer

    now = time.time()
    _frame_counter += 1

    # FPS measured on INPUT rate (all frames), not processing rate.
    # This shows the true camera/WS throughput visible to the user.
    dt = max(now - _last_frame_time, 0.001) if _last_frame_time > 0 else 0.033
    if _last_frame_time > 0:
        _fps_estimate = 0.7 * _fps_estimate + 0.3 / dt
    _last_frame_time = now

    # Frame dropping — skip frames to prevent pipeline backlog.
    # Processing takes ~45ms; frames arrive every ~40ms → queue grows.
    # Skipping every 2nd frame keeps throughput at 25fps with ~0 backlog.
    if _frame_counter % _input_skip != 0:
        # Return cached full result from last processed frame so the
        # frontend sees stable hands/gestures — no flickering.
        if _last_proc_result is not None:
            return _last_proc_result
        return {
            'fps': round(_fps_estimate, 1), 'detect_ms': 0, 'proc_ms': 0,
            'hands': [], 'hand_count': 0,
            'semantic': {'action': None, 'text': '', 'raw': [], 'buffer': '',
                         'mode': sentence_recorder.get_mode(),
                         'state': sentence_recorder.get_state()},
            'confidence': 0, 'frame_index': _frame_counter,
        }

    _video_timestamp_ms += int(dt * 1000 * _input_skip)  # compensate for skipped frames

    # Downscale to 128px — fastest setting balancing speed and accuracy
    h, w = image_bgr.shape[:2]
    if w > 128:
        scale = 128.0 / w
        image_small = cv2.resize(image_bgr, (128, int(h * scale)))
    else:
        image_small = image_bgr
    image_rgb_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

    t0 = time.time()
    # v3.2: First frame after reconnect uses IMAGE mode to avoid
    # MediaPipe VIDEO-mode timestamp corruption (internal state
    # from previous session may reject reset-to-zero timestamps).
    try:
        if _first_frame_after_reconnect:
            result = hand_tracker.detect_image(image_rgb_small)
            _first_frame_after_reconnect = False
        else:
            result = hand_tracker.detect_video(image_rgb_small, _video_timestamp_ms)
    except Exception as e:
        # Fallback: if VIDEO mode fails, try IMAGE mode and continue
        print(f'[Backend] detect_video failed ({e}), falling back to IMAGE mode')
        result = hand_tracker.detect_image(image_rgb_small)
    detect_ms = (time.time() - t0) * 1000
    proc_ms = 0.0  # computed at end of pipeline

    # v3.1: Optical flow for hand ROI (E4/E6 benefit from it, E3 ignores)
    if _frame_counter % _flow_skip == 0:
        gray_small = cv2.cvtColor(image_rgb_small, cv2.COLOR_RGB2GRAY)
        _cached_flow = flow_estimator.compute(gray_small)

    hands = []
    all_gestures = []
    max_palm_speed = 0.0
    all_stabilities = []

    if result.hand_landmarks:
        # v3.4: Collect raw hand data, then sort so Left is always first.
        # This ensures all_gestures[0] = left hand, all_gestures[1] = right hand.
        raw_hands = []
        for i in range(len(result.hand_landmarks)):
            wrapper = HandLandmarksWrapper(result.hand_landmarks[i])
            label_raw = result.handedness[i][0].category_name
            label = label_raw

            finger_up, finger_count, gesture, confidence = \
                rule_recognizer.recognize_with_confidence(wrapper)

            curr_kp = _landmarks_to_array(wrapper)
            prev_kp = _prev_keypoints.get(label)
            kin_feat = None
            if prev_kp is not None and _frame_counter % _kin_skip == 0:
                history = _kp_history.get(label, [])
                kin_feat = compute_all_kinematic_features(
                    prev_kp, curr_kp, fps=_fps_estimate, kp_history=history)
                if kin_feat['palm_speed'][0] > max_palm_speed:
                    max_palm_speed = kin_feat['palm_speed'][0]

            _prev_keypoints[label] = curr_kp
            if label not in _kp_history:
                _kp_history[label] = []
            _kp_history[label].append(curr_kp[:, :2])
            if len(_kp_history[label]) > _kinematic_history_len:
                _kp_history[label].pop(0)

            stability = float(kin_feat['trajectory_stability'][0]) if kin_feat else 0.5
            lm_list = [[round(lm.x, 3), round(lm.y, 3), round(lm.z, 3)]
                       for lm in wrapper.landmark]
            raw_hands.append({
                'label': label, 'landmarks': lm_list,
                'finger_up': finger_up, 'finger_count': finger_count,
                'gesture': gesture,
                'gesture_confidence': round(float(confidence), 2),
                'stability': round(stability, 2),
                'kin_feat': kin_feat,
            })

            # --- DL buffer stash (always include ROI+flow, E3 ignores them) ---
            if _current_model != 'rule' and _dl_available and _dl_buffer_lock is not None:
                buf = _dl_buffer_left if label == 'Left' else _dl_buffer_right
                roi_bgr = None
                if image_bgr is not None:
                    h_img, w_img = image_bgr.shape[:2]
                    xs = [int(curr_kp[j, 0] * w_img) for j in range(21)]
                    ys = [int(curr_kp[j, 1] * h_img) for j in range(21)]
                    pad = 25
                    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
                    x2, y2 = min(w_img, max(xs) + pad), min(h_img, max(ys) + pad)
                    if x2 > x1 and y2 > y1:
                        roi_bgr = cv2.resize(image_bgr[y1:y2, x1:x2], (96, 96))
                flow_hist = None
                if _cached_flow is not None:
                    from core.feature.motion_encoder import compute_flow_histogram
                    if x2 > x1 and y2 > y1 and _cached_flow.shape[0] > 0:
                        fh, fw = _cached_flow.shape[:2]
                        sx, sy = fw / w_img, fh / h_img
                        fx1, fy1 = int(x1 * sx), int(y1 * sy)
                        fx2, fy2 = int(x2 * sx), int(y2 * sy)
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(fw, fx2), min(fh, fy2)
                        if fx2 > fx1 and fy2 > fy1:
                            flow_crop = _cached_flow[fy1:fy2, fx1:fx2]
                            flow_hist = compute_flow_histogram(flow_crop, bins=64)
                    else:
                        flow_hist = np.zeros(128, dtype=np.float32)
                with _dl_buffer_lock:
                    buf.append({
                        'curr_kp': curr_kp.copy(),
                        'prev_kp': prev_kp.copy() if prev_kp is not None else None,
                        'roi_bgr': roi_bgr,
                        'flow_hist': flow_hist,
                    })

        # Sort: Left first
        raw_hands.sort(key=lambda h: 0 if h['label'] == 'Left' else 1)

        for h in raw_hands:
            all_stabilities.append(h['stability'])
            all_gestures.append(h['gesture'])
            hands.append({
                'label': h['label'], 'landmarks': h['landmarks'],
                'finger_up': h['finger_up'], 'finger_count': h['finger_count'],
                'gesture': h['gesture'],
                'gesture_confidence': h['gesture_confidence'],
                'stability': h['stability'],
            })

    # Route DL results to semantic pipeline (all_gestures) and UI (hands)
    dl_gesture_name = None
    dl_confidence_val = 0.0
    if _current_model != 'rule' and _dl_hand_results:
        now_ts = time.time()
        dl_gestures = []
        for side in ['Left', 'Right']:
            if side in _dl_hand_results:
                entry = _dl_hand_results[side]
                if len(entry) >= 3 and now_ts - entry[2] < 2.0 and entry[1] >= 30:
                    dl_gestures.append(entry[0])
        if len(dl_gestures) > 0:
            all_gestures = dl_gestures

        for h in hands:
            side = h['label']
            if side in _dl_hand_results:
                entry = _dl_hand_results[side]
                if len(entry) >= 3 and now_ts - entry[2] < 2.0 and entry[1] >= 50:
                    h['gesture'] = entry[0]
                    h['gesture_confidence'] = entry[1] / 100.0
        dl_gesture_name, dl_confidence_val = list(_dl_hand_results.values())[0][:2] if _dl_hand_results else (None, 0.0)

    # v3.1: Sentence recorder — only process every _sr_skip frames or on gesture change
    current_gestures = tuple(all_gestures) if all_gestures else ()
    sr_result = _last_sr_result
    if _frame_counter % _sr_skip == 0 or not result.hand_landmarks:
        primary_gesture = all_gestures[0] if all_gestures else 'None'
        hand_count = len(all_gestures)
        sr_result = sentence_recorder.process(
            primary_gesture, hand_count, all_gestures,
            kinematic_features=None)
        _last_sr_result = sr_result

    semantic_action = None
    semantic_text = ''
    raw_tokens = []
    buffer_display = ''
    mode_name = sentence_recorder.get_mode()

    if sr_result:
        action = sr_result[0]
        if action == 'start':
            semantic_action = 'start'
            # Reset perf counters for new session
            _perf_fps_sum = _perf_detect_sum = _perf_proc_sum = 0.0
            _perf_frame_count = 0
            _perf_session_start = time.time()
            _perf_snapshot = False
            global _perf_dl_conf_sum, _perf_dl_conf_count
            _perf_dl_conf_sum = _perf_dl_conf_count = 0
        elif action == 'output':
            semantic_action = 'output'
            semantic_text = sr_result[1]
            raw_tokens = sr_result[2] if len(sr_result) > 2 else []
            _perf_snapshot = True
        elif action == 'buffer':
            semantic_action = 'buffer'
            buffer_display = ' -> '.join(sr_result[1])
            raw_tokens = sr_result[1]
        elif action == 'undo_word':
            semantic_action = 'undo_word'
            buffer_display = ' -> '.join(sr_result[2]) if len(sr_result) > 2 else ''
            raw_tokens = sr_result[2] if len(sr_result) > 2 else []
        elif action == 'undo_sentence':
            semantic_action = 'undo_sentence'
        elif action == 'separator':
            semantic_action = 'separator'

    # v3.1: Only force-end if actually recording + buffer has content + hands lost for > 4s.
    if not result.hand_landmarks and sentence_recorder.get_state() == 'RECORDING':
        timed_out = semantic_parser.check_timeout()
        if timed_out:
            sr_result = sentence_recorder.force_end()
            if sr_result:
                semantic_action = 'output'
                semantic_text = sr_result[1] if len(sr_result) > 1 else ''
                raw_tokens = sr_result[2] if len(sr_result) > 2 else []

    # Prevent stale _last_sr_result from being re-processed
    if semantic_action in ('output', 'start'):
        _last_sr_result = None

    # Auto-save performance log on session end
    if _perf_snapshot and _perf_frame_count > 0:
        import json as _json2, os as _os2
        n = max(_perf_frame_count, 1)
        entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_s': round(time.time() - _perf_session_start, 1),
            'frames': _perf_frame_count,
            'avg_fps': round(_perf_fps_sum / n, 1),
            'avg_detect_ms': round(_perf_detect_sum / n, 1),
            'avg_proc_ms': round(_perf_proc_sum / n, 1),
            'output': semantic_text,
            'mode': _current_model.upper() if _current_model != 'rule' else 'Rule',
            'dl_avg_conf': round(_perf_dl_conf_sum / max(_perf_dl_conf_count, 1), 2),
            'dl_conf_count': _perf_dl_conf_count,
        }
        log_path = 'data/performance_log.json'
        if _os2.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                try: history = _json2.load(f)
                except: history = []
        else:
            history = []
        history.append(entry)
        if len(history) > 200:
            history = history[-100:]
        with open(log_path, 'w', encoding='utf-8') as f:
            _json2.dump(history, f, ensure_ascii=False, indent=2)
        _avg = entry['avg_fps']; _det = entry['avg_detect_ms']; _proc = entry['avg_proc_ms']
        print(f'[Perf] Session saved: {_avg}fps, det {_det}ms, proc {_proc}ms')
        _perf_snapshot = False

    avg_stability = sum(all_stabilities) / len(all_stabilities) if all_stabilities else 0.0
    proc_ms = round((time.time() - now) * 1000, 1)

    # Accumulate performance stats while recording (after proc_ms computed)
    if sentence_recorder.get_state() == 'RECORDING':
        _perf_fps_sum += _fps_estimate
        _perf_detect_sum += detect_ms
        _perf_proc_sum += proc_ms
        _perf_frame_count += 1

    proc_result = {
        'fps': round(_fps_estimate, 1),
        'detect_ms': round(detect_ms, 1),
        'proc_ms': proc_ms,
        'hands': hands,
        'hand_count': len(all_gestures),
        'semantic': {
            'action': semantic_action,
            'text': semantic_text,
            'raw': raw_tokens,
            'buffer': buffer_display,
            'mode': mode_name,
            'state': sentence_recorder.get_state(),
        },
        'confidence': round(avg_stability * 100),
        'frame_index': _frame_counter,
        'model': _current_model,
        'dl_gesture': dl_gesture_name,
        'dl_confidence': dl_confidence_val,
    }
    # Debounce: only update cached result when hand count is stable.
    # Prevents single-frame dropouts from causing flicker on skipped frames.
    # Strip transient actions from cached copy so skipped frames don't
    # re-trigger frontend opLog pushes. Only 'buffer' and None persist.
    cached = dict(proc_result)
    cached['semantic'] = dict(proc_result['semantic'])
    if semantic_action in ('output', 'start', 'separator', 'undo_word', 'undo_sentence'):
        cached['semantic']['action'] = None

    curr_hand_count = len(proc_result.get('hands', []))
    if curr_hand_count >= _last_hand_count:
        _last_hand_count = curr_hand_count
        _hand_debounce_timer = 0
        _last_proc_result = cached
    else:
        _hand_debounce_timer += 1
        if _hand_debounce_timer >= 3:
            _last_hand_count = curr_hand_count
            _hand_debounce_timer = 0
            _last_proc_result = cached
    return proc_result


# ---- API Routes ----

@app.get('/api/health')
def health():
    return {
        'status': 'ok', 'version': '6.0',
        'deepseek_available': deepseek_client.is_available() if deepseek_client else False,
    }


@app.post('/api/shutdown')
def shutdown():
    """Gracefully shut down the backend server."""
    print('[Backend] Shutdown requested, exiting...')
    import os as _os
    _os._exit(0)


@app.post('/api/set_api_key')
def set_api_key(data: SetApiKeyRequest):
    """Set or update DeepSeek API key at runtime. Key is stored in memory only."""
    global deepseek_client, sentence_recorder
    api_key = data.api_key.strip()
    if not api_key:
        return {'ok': False, 'error': 'api_key is required'}
    if not api_key.startswith('sk-'):
        return {'ok': False, 'error': 'invalid key format (must start with sk-)'}

    from core.semantic.deepseek_client import DeepSeekClient
    deepseek_client = DeepSeekClient(api_key=api_key)
    sentence_recorder.deepseek = deepseek_client
    print(f'[Backend] DeepSeek API key updated (available={deepseek_client.is_available()})')
    return {'ok': True, 'deepseek_available': deepseek_client.is_available()}


@app.post('/api/clear_api_key')
def clear_api_key():
    """Clear the DeepSeek API key from memory."""
    global deepseek_client, sentence_recorder
    deepseek_client = DeepSeekClient()
    sentence_recorder.deepseek = deepseek_client
    print('[Backend] DeepSeek API key cleared')
    return {'ok': True, 'deepseek_available': False}


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    global _prev_keypoints, _kp_history, _video_timestamp_ms, _last_sr_result, _perf_snapshot
    global _last_frame_time, _first_frame_after_reconnect, _last_proc_result
    global _current_model, _dl_buffer_left, _dl_buffer_right, _dl_hand_results, _last_hand_count, _hand_debounce_timer
    global _work_mode, _triage_history
    await ws.accept()
    print('[WS] Client connected')

    # v3.2: Full state reset on new connection
    sentence_recorder.cancel()
    _prev_keypoints = {}
    _kp_history = {}
    _video_timestamp_ms = 0
    _last_frame_time = 0
    _last_sr_result = None
    _last_proc_result = None
    _first_frame_after_reconnect = True
    if _dl_buffer_left is not None:
        _dl_buffer_left.clear()
        _dl_buffer_right.clear()
    _dl_hand_results.clear()
    _last_hand_count = 0
    _hand_debounce_timer = 0

    frame_count = 0
    try:
        while True:
            msg = await ws.receive()
            # --- Text message: control command ---
            if 'text' in msg:
                try:
                    cmd = json.loads(msg['text'])
                    action = cmd.get('action', '')
                    if action == 'toggle_dl':
                        # Cycle: rule → e3 → e4 → e6 → rule
                        cycle = {'rule': 'e3', 'e3': 'e4', 'e4': 'e6', 'e6': 'rule'}
                        _current_model = cycle.get(_current_model, 'e6')
                        _dl_buffer_left.clear()
                        _dl_buffer_right.clear()
                        _dl_hand_results.clear()
                        print(f'[WS] Model cycled to: {_current_model}')
                        await ws.send_json({'mode_changed': True, 'model': _current_model})
                    elif action == 'set_model':
                        model = cmd.get('model', 'e6')
                        if model in ('e3', 'e4', 'e6', 'rule'):
                            _current_model = model
                            _dl_buffer_left.clear()
                            _dl_buffer_right.clear()
                            _dl_hand_results.clear()
                            print(f'[WS] Model set to: {_current_model}')
                            await ws.send_json({'mode_changed': True, 'model': _current_model})
                    elif action == 'force_end':
                        if sentence_recorder.get_state() == 'RECORDING':
                            sr = sentence_recorder.force_end()
                            if sr:
                                _last_sr_result = sr
                                _perf_snapshot = True  # trigger save on next output
                    elif action == 'set_work_mode':
                        wm = cmd.get('work_mode', 'translate')
                        _work_mode = wm
                        _triage_history = []
                        print(f'[WS] Work mode: {wm}')
                        await ws.send_json({'work_mode_changed': True, 'work_mode': _work_mode})
                    elif action == 'swap_hands':
                        _swap_hands = not _swap_hands
                        sentence_recorder._swap_hands = _swap_hands
                        print(f'[WS] Swap hands: {_swap_hands}')
                        await ws.send_json({'hands_swapped': True, 'swap_hands': _swap_hands})
                except json.JSONDecodeError:
                    pass
                continue

            # --- Binary message: JPEG frame ---
            data = msg.get('bytes')
            if data is None:
                continue

            frame_count += 1
            # Decode JPEG frame
            img_array = np.frombuffer(data, dtype=np.uint8)
            image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue

            # --- DeepSeek blocks event loop if called synchronously ---
            # Disable during _process_frame, then run in executor below
            _ds = sentence_recorder.deepseek
            sentence_recorder.deepseek = None
            proc_result = _process_frame(image_bgr)
            sentence_recorder.deepseek = _ds
            await ws.send_json(_py(proc_result))

            # --- Translate mode: non-blocking DeepSeek polish ---
            if (_work_mode == 'translate' and
                  proc_result.get('semantic', {}).get('action') == 'output' and
                  proc_result.get('semantic', {}).get('text') and
                  _ds and _ds.is_available()):
                raw_tokens = proc_result['semantic'].get('raw', [])
                base_text = proc_result['semantic']['text']
                try:
                    loop = asyncio.get_event_loop()
                    polished = await loop.run_in_executor(
                        None, _ds.reorder_and_assemble,
                        [t for t in raw_tokens if t and t != 'None' and t != '__SEP__'],
                        'common', sentence_recorder.get_mode(), [])
                    if not polished:
                        polished = await loop.run_in_executor(
                            None, _ds.hospital_polish, base_text)
                    if polished and polished != base_text:
                        await ws.send_json(_py({
                            'polish': polished,
                            'semantic': {'action': 'polish', 'text': polished},
                        }))
                except Exception:
                    pass

            # --- Triage mode: non-blocking DeepSeek conversation ---
            if (_work_mode == 'triage' and
                  proc_result.get('semantic', {}).get('action') == 'output' and
                  proc_result.get('semantic', {}).get('text')):
                patient_text = proc_result['semantic']['text']
                try:
                    import asyncio as _asyncio
                    loop = _asyncio.get_event_loop()
                    triage_resp = await loop.run_in_executor(
                        None, deepseek_client.triage_conversation,
                        patient_text, list(_triage_history))
                    if triage_resp:
                        _triage_history.append({'role': 'patient', 'content': patient_text})
                        _triage_history.append({'role': 'assistant', 'content': triage_resp['message']})
                        if len(_triage_history) > 20:
                            _triage_history = _triage_history[-20:]
                        await ws.send_json(_py({
                            'triage': triage_resp,
                            'triage_history': _triage_history[-6:],
                        }))
                except Exception as e:
                    print(f'[WS] Triage error: {e}')

            if frame_count % 120 == 0:
                hands_detected = len(proc_result.get('hands', []))
                dms = proc_result.get('detect_ms', 0)
                pms = proc_result.get('proc_ms', 0)
                print(f'[WS] F{frame_count}: {proc_result["fps"]:.0f}fps {hands_detected}h {dms:.0f}ms proc {pms:.0f}ms [{_current_model.upper()}]')

    except WebSocketDisconnect:
        print('[WS] Client disconnected')
    except Exception as e:
        print(f'[WS] Error: {e}')


@app.get('/')
def root():
    return FileResponse('frontend/index.html')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
