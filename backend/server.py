# coding:utf-8
"""FastAPI WebSocket backend for hospital sign language input method v3.0.

Architecture:
  Vue frontend (camera + UI) <--WebSocket--> FastAPI backend (recognition pipeline)

Receives frames as JPEG bytes, returns recognition results as JSON.
Stateless per-frame design — SentenceRecorder state is maintained on the server.
"""
import sys, os, json, time, base64, io, asyncio
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
import uvicorn

# ---- Init recognition pipeline ----
from core.perception.hand_tracker import HandTracker, HandLandmarksWrapper
from core.perception.flow_estimator import FlowEstimator
from core.fallback.rule_recognizer import RuleRecognizer
from core.semantic.temporal_parser import TemporalSemanticParser
from core.semantic.sentence_recorder import SentenceRecorder
from core.semantic.deepseek_client import DeepSeekClient
from core.feature.kinematic_features import compute_all_kinematic_features

# ---- Init DL pipeline (optional, graceful fallback) ----
_dl_available = False
_spatial_encoder = None
_motion_encoder = None
_fusion_model = None
_onnx_recognizer = None
_dl_buffer_left = None   # deque for left hand features
_dl_buffer_right = None  # deque for right hand features
_dl_hand_results = {}    # {'Left': (name, conf), 'Right': (name, conf)}
_dl_gesture_names = {}   # idx → name mapping

try:
    import torch
    from collections import deque
    from core.feature.spatial_gcn import SpatialGCN, normalize_keypoints, extract_hand_angles
    from core.feature.motion_encoder import MotionEncoder, extract_motion_features
    from core.feature.multimodal_fusion import CrossModalFusion
    from core.feature.visual_encoder import LightweightVisualEncoder
    from core.inference.onnx_recognizer import ONNXRecognizer

    _spatial_encoder = SpatialGCN()
    _spatial_encoder.eval()
    _motion_encoder = MotionEncoder()
    _motion_encoder.eval()
    _fusion_model = CrossModalFusion()
    _fusion_model.eval()
    _visual_encoder = LightweightVisualEncoder()
    _visual_encoder.eval()
    _onnx_recognizer = ONNXRecognizer()
    _dl_buffer_left = deque(maxlen=32)
    _dl_buffer_right = deque(maxlen=32)
    _dl_available = True
    print('[Backend] DL pipeline loaded (SpatialGCN + Motion + Visual + Fusion + TCN)')
except ImportError as e:
    print(f'[Backend] DL pipeline not available (missing torch/onnx): {e}')
except Exception as e:
    print(f'[Backend] DL pipeline init failed: {e}')

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
_flow_skip = 999           # optical flow disabled — not used by rule path, expensive
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
_kinematic_history_len = 16
_kin_skip = 6              # kinematic features every 6 processed frames
_sr_skip = 1               # sentence recorder on every processed frame (input_skip already gates)
_last_sr_result = None
_last_proc_result = None   # cached full proc_result for skipped frames
_last_hand_count = 0       # debounce flicker: track consistent hand count
_hand_debounce_timer = 0
_use_dl = False            # True = deep learning, False = rule-based
_work_mode = 'translate'   # 'translate' = sign language translation, 'triage' = AI triage
_swap_hands = False        # v3.4: False=left mode right content, True=swapped
_triage_history = []       # conversation history for triage mode
_video_timestamp_ms = 0    # v3.1: increasing timestamp for VIDEO mode tracking
_first_frame_after_reconnect = True  # v3.2: IMAGE mode on first frame after WS reconnect

print('[Backend] Pipeline ready.')


def _extract_dl_features(curr_kp, prev_kp, hand_landmarks_wrapper=None, frame_bgr=None):
    """Extract 256-dim fused feature vector for one frame.

    Uses PyTorch SpatialGCN + MotionEncoder + VisualEncoder + CrossModalFusion.
    Returns [256] numpy array, or None on failure.
    """
    if not _dl_available:
        return None
    try:
        with torch.no_grad():
            curr_t = torch.from_numpy(curr_kp).float().unsqueeze(0)  # [1, 21, 3]
            spatial = _spatial_encoder(curr_t)                        # [1, 256]

            if prev_kp is not None:
                prev_t = torch.from_numpy(prev_kp).float()
                kp_diff = (curr_t[:, :, :2] - prev_t[:21, :2].unsqueeze(0)).reshape(1, 42)
                kp_delta = torch.cat([kp_diff, torch.zeros(1, 21)], dim=1)  # [1, 63]
                flow_hist = torch.zeros(1, 128)
            else:
                kp_delta = torch.zeros(1, 63)
                flow_hist = torch.zeros(1, 128)

            motion = _motion_encoder(flow_hist, kp_delta)            # [1, 128]

            # Visual: crop 96x96 hand ROI from frame
            if frame_bgr is not None:
                h, w = frame_bgr.shape[:2]
                # Get landmark pixel coords from keypoints or wrapper
                if hand_landmarks_wrapper is not None:
                    xs = [int(lm.x * w) for lm in hand_landmarks_wrapper.landmark]
                    ys = [int(lm.y * h) for lm in hand_landmarks_wrapper.landmark]
                else:
                    xs = [int(curr_kp[j, 0] * w) for j in range(21)]
                    ys = [int(curr_kp[j, 1] * h) for j in range(21)]
                pad = 25
                x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
                x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
                if x2 > x1 and y2 > y1:
                    roi_bgr = cv2.resize(frame_bgr[y1:y2, x1:x2], (96, 96))
                    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                    roi_t = torch.from_numpy(roi_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    visual = _visual_encoder(roi_t)  # [1, 512]
                else:
                    visual = torch.zeros(1, 512)
            else:
                visual = torch.zeros(1, 512)

            fused = _fusion_model(visual, spatial, motion)           # [1, 256]
            return fused.squeeze(0).numpy().astype(np.float32)
    except Exception:
        return None


def _dl_process():
    """Background daemon: PyTorch feature extraction + ONNX inference."""
    import time as _t
    while True:
        _t.sleep(2.0)
        if not _use_dl or not _dl_available:
            continue
        try:
            for side, buf in [('Left', _dl_buffer_left), ('Right', _dl_buffer_right)]:
                if len(buf) < 6:
                    continue
                items = list(buf)[-16:]
                kps = np.stack([it[0] for it in items])
                with torch.no_grad():
                    kp_t = torch.from_numpy(kps).float()  # [T, 21, 3]
                    spatial = _spatial_encoder(kp_t)
                    T = kp_t.shape[0]
                    feats = []
                    for t in range(T):
                        mo = torch.zeros(1, 128)
                        if t > 0 and items[t][1] is not None:
                            prev = torch.from_numpy(items[t][1]).float()
                            kd = (kp_t[t:t+1,:,:2] - prev[:21,:2].unsqueeze(0)).reshape(1, 42)
                            mo = _motion_encoder(torch.zeros(1, 128), torch.cat([kd, torch.zeros(1, 21)], dim=1))
                        fu = _fusion_model(torch.zeros(1, 512), spatial[t:t+1], mo)
                        feats.append(fu.squeeze(0).numpy().astype(np.float32))
                if len(feats) >= 6:
                    result = _onnx_recognizer.predict_sequence_label(np.stack(feats))
                    if result is not None:
                        lid, conf = result
                        _dl_hand_results[side] = (_dl_gesture_names.get(lid, str(lid)), round(float(conf) * 100))
                buf.clear()
        except Exception:
            pass

import threading as _th
_th.Thread(target=_dl_process, daemon=True).start()

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

    # v3.1: Optical flow only every N frames (expensive, not used by rule path)
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

    # --- DL: non-blocking via background timer ---
    dl_gesture_name = None
    dl_confidence_val = 0.0
    if _use_dl and _dl_available and result.hand_landmarks and _frame_counter % 30 == 0:
        # Just stash raw keypoints — zero PyTorch in main thread
        for i in range(len(result.hand_landmarks)):
            label = result.handedness[i][0].category_name
            curr_kp = _landmarks_to_array(HandLandmarksWrapper(result.hand_landmarks[i]))
            prev_kp = _prev_keypoints.get(label)
            buf = _dl_buffer_left if label == 'Left' else _dl_buffer_right
            buf.append((curr_kp.copy(), prev_kp.copy() if prev_kp is not None else None))

    # Display latest DL results in hand cards
    if _use_dl and _dl_hand_results:
        for h in hands:
            side = h['label']
            if side in _dl_hand_results:
                name, conf = _dl_hand_results[side]
                h['gesture'] = name
                h['gesture_confidence'] = conf / 100.0
        dl_gesture_name, dl_confidence_val = list(_dl_hand_results.values())[0] if _dl_hand_results else (None, 0.0)

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
            'mode': 'DL' if _use_dl else 'Rule',
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
        'use_dl': _use_dl,
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


def _get_bbox(wrapper, shape):
    h, w = shape[:2]
    xs = [int(lm.x * w) for lm in wrapper.landmark]
    ys = [int(lm.y * h) for lm in wrapper.landmark]
    return [min(xs), min(ys), max(xs), max(ys)]


# ---- API Routes ----

@app.get('/api/health')
def health():
    return {'status': 'ok', 'version': '3.0'}


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    global _prev_keypoints, _kp_history, _video_timestamp_ms, _last_sr_result, _perf_snapshot
    global _last_frame_time, _first_frame_after_reconnect, _last_proc_result
    global _use_dl, _dl_buffer_left, _dl_buffer_right, _dl_hand_results, _last_hand_count, _hand_debounce_timer
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
                        _use_dl = not _use_dl
                        if _use_dl:
                            _dl_buffer_left.clear()
                            _dl_buffer_right.clear()
                        _dl_hand_results.clear()
                        mode_name = 'DL' if _use_dl else 'Rule'
                        print(f'[WS] Mode switched to: {mode_name}')
                        await ws.send_json({'mode_changed': True, 'use_dl': _use_dl})
                    elif action == 'set_mode':
                        mode = cmd.get('mode', 'rule')
                        _use_dl = (mode == 'dl')
                        if _use_dl:
                            _dl_buffer_left.clear()
                            _dl_buffer_right.clear()
                        _dl_hand_results.clear()
                        print(f'[WS] Mode set to: {mode}')
                        await ws.send_json({'mode_changed': True, 'use_dl': _use_dl})
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
                mode_tag = 'DL' if _use_dl else 'Rule'
                print(f'[WS] F{frame_count}: {proc_result["fps"]:.0f}fps {hands_detected}h {dms:.0f}ms proc {pms:.0f}ms [{mode_tag}]')

    except WebSocketDisconnect:
        print('[WS] Client disconnected')
    except Exception as e:
        print(f'[WS] Error: {e}')


@app.get('/')
def root():
    return FileResponse('frontend/index.html')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
