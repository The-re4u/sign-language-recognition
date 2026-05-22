# coding:utf-8
"""
Sentence-level gesture recorder v3.4 — hospital sign language input method.

Design principle: 7 mode gestures × 10 content gestures = 70 semantic slots.
Some slots are reserved for control functions (separator, undo).
  - Mode gestures {Good, Seven, Victory, Eight, Two, Pinky_Up, One}
  - Content gestures {Closed_Fist, One, Two, Three, Four, Open_Palm, Six, Seven, Eight, Nine}
  - One, Two, Seven, Eight serve dual roles (mode or content depending on context)

Control gestures use SYMMETRIC dual-gesture:
  - 双手 Victory (持握0.5s)  → undo last word
  - 双手 Seven (持握0.5s)    → undo sentence
  - 双手 One (持握0.5s)      → separator

START/END:
  - START: both hands Open_Palm sustained 1.0s
  - END:   both hands Closed_Fist sustained 1.5s, or transition Open→Fist
"""
import time
from collections import deque


class SentenceRecorder:

    # ================================================================
    # Mode gestures (non-dominant hand)
    # ================================================================
    MODE_NUMBER   = 'Good'         # 👍 → digit layer
    MODE_BODY     = 'Seven'        # 🤟 → body part layer
    MODE_HOSPITAL = 'Victory'      # ✌️ → hospital operations layer
    MODE_SYMPTOM  = 'Eight'        # 👍👆 → symptom layer
    MODE_STATUS   = 'Two'          # ✌️ → severity / status layer
    MODE_TIME     = 'Pinky_Up'     # 🤙 → time / frequency layer

    MODE_GESTURES = {MODE_NUMBER, MODE_BODY, MODE_HOSPITAL, MODE_SYMPTOM,
                     MODE_STATUS, MODE_TIME, 'One'}

    MODE_NAMES = {
        'Good': '数字', 'Seven': '身体部位',
        'Victory': '医院操作', 'Eight': '病症',
        'Two': '程度状态', 'Pinky_Up': '时间频次',
        'One': '常用语',
    }

    # ================================================================
    # Content gestures (right hand) — 10 gestures for 0-9 mapping
    # ================================================================
    CONTENT_GESTURES = {
        'Closed_Fist', 'One', 'Two', 'Three', 'Four',
        'Open_Palm', 'Six', 'Seven', 'Eight', 'Nine'
    }

    # ================================================================
    # Semantic maps: mode → (content_gesture → Chinese meaning)
    # 10 content gestures per layer:
    #   Closed_Fist, One, Two, Three, Four, Open_Palm, Six, Seven, Eight, Nine
    # ================================================================
    MAP_NUMBER = {
        'Closed_Fist': '0', 'One': '1', 'Two': '2', 'Three': '3', 'Four': '4',
        'Open_Palm': '5', 'Six': '6', 'Seven': '7', 'Eight': '8', 'Nine': '9',
    }

    MAP_BODY = {
        'Closed_Fist': '头', 'One': '眼', 'Two': '耳', 'Three': '背', 'Four': '腿',
        'Open_Palm': '手', 'Six': '脚', 'Seven': '喉', 'Eight': '胸', 'Nine': '胃',
    }

    MAP_HOSPITAL = {
        'Closed_Fist': '挂号', 'One': '取药', 'Two': '住院', 'Three': '输液', 'Four': '检查',
        'Open_Palm': '出院', 'Six': '转科', 'Seven': '化验', 'Eight': '手术', 'Nine': '换药',
    }

    MAP_SYMPTOM = {
        'Closed_Fist': '疼', 'One': '咳嗽', 'Two': '胸闷', 'Three': '食欲不振', 'Four': '腹泻',
        'Open_Palm': '发烧', 'Six': '恶心', 'Seven': '头晕', 'Eight': '乏力', 'Nine': '呼吸困难',
    }

    MAP_STATUS = {
        'Closed_Fist': '无', 'One': '轻微', 'Two': '中等', 'Three': '较重', 'Four': '严重',
        'Open_Palm': '紧急', 'Six': '好转', 'Seven': '恶化', 'Eight': '稳定', 'Nine': '反复',
    }

    MAP_TIME = {
        'Closed_Fist': '秒', 'One': '昨天', 'Two': '今天', 'Three': '分钟', 'Four': '小时',
        'Open_Palm': '天', 'Six': '现在', 'Seven': '明天', 'Eight': '持续', 'Nine': '早上',
    }

    MAP_PHRASE = {
        'Closed_Fist': '请帮帮我',
        'One':        ',',
        'Two':        '好的',
        'Three':      '谢谢',
        'Four':       '我想要',
        'Open_Palm':  '你好',
        'Six':        '是',
        'Seven':      '再见',
        'Eight':      '我不舒服',
        'Nine':       '不是',
    }

    MODE_MAPS = {
        'Good': MAP_NUMBER,
        'Seven': MAP_BODY,
        'Victory': MAP_HOSPITAL,
        'Eight': MAP_SYMPTOM,
        'Two': MAP_STATUS,
        'Pinky_Up': MAP_TIME,
        'One': MAP_PHRASE,
    }

    # ================================================================
    # Control: symmetric dual-gesture (both hands same mode gesture).
    # Semantically safe because normal input always uses one mode hand
    # + one content hand — two hands with the same mode gesture never
    # occurs during valid semantic input.
    # ================================================================
    CTRL_UNDO_WORD     = frozenset({'Victory'})
    CTRL_UNDO_SENTENCE = frozenset({'Seven'})
    CTRL_SEPARATOR     = frozenset({'One'})

    # ================================================================
    # Transition filter
    # ================================================================
    TRANSITION_DEAD_ZONE = 0.3
    COORD_CHANGE_THRESHOLD = 0.15
    PALM_SPEED_THRESHOLD = 0.05

    def __init__(self, semantic_parser, cooldown=0.35, timeout=15.0,
                 deepseek_client=None, min_gesture_duration=0.3):
        self.semantic_parser = semantic_parser
        self.cooldown = cooldown
        self.timeout = timeout
        self.min_gesture_duration = min_gesture_duration
        self.deepseek = deepseek_client

        self.state = 'IDLE'
        self.buffer = []
        self._last_buffer_mode = None
        self.last_gesture_time = 0
        self.last_cooldown_end = 0
        self.last_gesture = None
        self.last_gesture_change_time = 0
        self._pending_gesture = None
        self._pending_gesture_start = 0
        self._current_mode = None        # current mode gesture name
        self._current_mode_name = '基础'   # human-readable mode name
        self.sentence_history = []
        self._prev_dual_state = None
        self._idle_open_start = 0     # v3.0: fallback START timer
        self._recording_fist_start = 0  # v3.0: fallback END timer
        self._single_fist_start = 0    # v3.2: single-hand fist END timer
        self._last_raw_tokens = []      # v3.0: for UI history display
        self._ctrl_pair = None          # v3.2: sustained asymmetric-control tracking
        self._ctrl_start = 0            # v3.2: control hold start time
        self._last_separator_time = 0   # v3.2: separator cooldown to prevent flicker
        self._last_start_time = 0       # v3.2: prevent duplicate start within 1s
        self._last_buffer_mode = None   # v3.4: auto-separator on mode change
        self._swap_hands = False        # v3.4: False=L mode R content, True=swapped

    # ================================================================
    # Public API
    # ================================================================

    def process(self, gesture, hand_count=1, all_gestures=None,
                kinematic_features=None):
        """Process incoming gesture with clean dual-hand protocol.

        Returns action tuple or None.
        """
        now = time.time()

        # Compute dual-hand state FIRST — used as control gate
        curr_dual = self._dual_state(all_gestures)

        # Transition filter (only meaningful outside control states)
        if kinematic_features and self._is_transition(kinematic_features):
            return ('status', 'RECORDING')

        # --- START/END: transition-based ---
        start_trigger = (self._prev_dual_state == 'both_fist' and
                         curr_dual == 'both_open')
        end_trigger = (self._prev_dual_state == 'both_open' and
                       curr_dual == 'both_fist')
        self._prev_dual_state = curr_dual

        if self.state == 'IDLE':
            if now - self._last_start_time < 1.0:
                return None
            # Primary: transition fist→open
            if start_trigger:
                return self._start_recording()
            # Fallback: sustained dual Open_Palm for 0.5s also starts
            if curr_dual == 'both_open':
                if self._idle_open_start == 0:
                    self._idle_open_start = now
                elif now - self._idle_open_start > 0.5:
                    self._idle_open_start = 0
                    return self._start_recording()
            else:
                self._idle_open_start = 0
            return None

        if self.state == 'RECORDING':
            # --- CONTROL GATE: both_fist → end recording ---
            if curr_dual == 'both_fist':
                # v3.2: Transition-based end: Open_Palm → Closed_Fist, requires 1.0s gap
                if end_trigger and (now - self.last_gesture_time) > 1.0:
                    self._recording_fist_start = 0
                    return self._end_recording()
                # Fallback: sustained dual Closed_Fist for 1.5s
                if self._recording_fist_start == 0:
                    self._recording_fist_start = now
                elif now - self._recording_fist_start > 1.5:
                    self._recording_fist_start = 0
                    return self._end_recording()
                # Skip content while both_fist (pending end)
                return None

            # Reset fist timer when not in control state
            self._recording_fist_start = 0

            # --- Control: asymmetric dual-gesture controls ---
            ctrl = self._check_dual_control(all_gestures, now)
            if ctrl is not None:
                return ctrl

            # Single-hand END: only one hand detected, it's Closed_Fist, sustained 2.0s (was 3.5s)
            if (all_gestures and len(all_gestures) == 1
                  and all_gestures[0] == 'Closed_Fist'):
                if self._single_fist_start == 0:
                    self._single_fist_start = now
                elif now - self._single_fist_start > 2.0:
                    self._single_fist_start = 0
                    return self._end_recording()
            else:
                self._single_fist_start = 0

            # Auto-end on idle timeout — requires at least 1 gesture in buffer
            if self.last_gesture_time > 0 and len(self.buffer) >= 1 and \
               (now - self.last_gesture_time) > self.timeout:
                return self._end_recording()

            # --- Content processing ---
            # Classify hands → mode gesture + content gesture
            mode_gesture, content_gesture = self._classify_hands(all_gestures)

            # Resolve content → Chinese meaning based on mode
            resolved = self._resolve(content_gesture, mode_gesture)

            if now < self.last_cooldown_end:
                self._pending_gesture = None
                return ('status', 'RECORDING')

            # Gesture duration threshold
            min_dur = self.min_gesture_duration
            if resolved and resolved != 'None' and resolved != self.last_gesture:
                if self._pending_gesture != resolved:
                    self._pending_gesture = resolved
                    self._pending_gesture_start = now
                    self.last_gesture_change_time = now
                elif now - self._pending_gesture_start >= min_dur:
                    # v3.4: Auto-separator when mode layer changes
                    if (self._last_buffer_mode is not None and
                          mode_gesture is not None and
                          mode_gesture != self._last_buffer_mode and
                          len(self.buffer) > 0):
                        self.buffer.append(('，', now))
                    self.buffer.append((resolved, now))
                    self._last_buffer_mode = mode_gesture
                    self.last_gesture = resolved
                    self.last_gesture_time = now
                    self.last_cooldown_end = now + self.cooldown
                    self._pending_gesture = None
                    return ('buffer', [g for g, _ in self.buffer])
            elif resolved and resolved == self.last_gesture:
                self._pending_gesture = None

        return None

    # ================================================================
    # Hand classification
    # ================================================================

    def _classify_hands(self, all_gestures):
        """v3.4: Left hand = mode, right hand = content (or swapped).
        Hands are pre-sorted by server: all_gestures[0]=Left, all_gestures[1]=Right.
        Returns (mode_gesture, content_gesture).
        """
        # Persistent mode expires after 1.5s of no two-hand detection
        if self._current_mode is not None:
            if time.time() - self.last_gesture_time > 1.5:
                self._current_mode = None
                self._current_mode_name = '基础'

        if not all_gestures or len(all_gestures) < 2:
            g = all_gestures[0] if all_gestures else 'None'
            # Single-hand: use persistent mode if set
            if self._current_mode is not None:
                return (self._current_mode, g)
            return (None, None)

        left_g, right_g = all_gestures[0], all_gestures[1]

        # Both same gesture: suppress (dual-control or start/end transition)
        if left_g == right_g:
            return (None, None)

        # Hardcoded: left=mode, right=content (or swapped via UI toggle)
        if self._swap_hands:
            mode_g, content_g = right_g, left_g
        else:
            mode_g, content_g = left_g, right_g

        # Only update persistent mode if mode hand shows a known mode gesture
        if mode_g in self.MODE_GESTURES:
            self._current_mode = mode_g
            self._current_mode_name = self.MODE_NAMES.get(mode_g, '?')

        return (mode_g, content_g)

    def _resolve(self, content, mode):
        """Map content gesture to Chinese meaning based on mode layer."""
        if content is None or content == 'None':
            return 'None'
        if mode and mode in self.MODE_MAPS:
            resolved = self.MODE_MAPS[mode].get(content)
            if resolved is not None:
                return resolved
        # No mode or content not mapped in this layer → suppress
        return 'None'

    # ================================================================
    # Control gestures
    # ================================================================

    def _check_dual_control(self, all_gestures, now):
        """Check for symmetric dual-gesture control patterns.

        Controls fire when both hands show the SAME mode gesture
        (e.g. both Good = undo word). This never overlaps with
        semantic input because normal mode+content always uses
        one mode hand and one content hand.
        """
        if not all_gestures or len(all_gestures) < 2:
            self._ctrl_pair = None; return None
        if self.state != 'RECORDING':
            self._ctrl_pair = None; return None

        pair = frozenset(all_gestures)

        # Both gestures must be mode gestures (not content)
        if not pair.issubset(self.MODE_GESTURES):
            self._ctrl_pair = None; return None

        # Must be symmetric — both hands doing the SAME gesture
        if len(pair) != 1:
            self._ctrl_pair = None; return None

        # Sustained hold: block content processing during control hold.
        # Returning a non-None sentinel prevents _classify_hands / _resolve
        # from treating the mode gesture as content and leaking it into buffer.
        if pair != self._ctrl_pair:
            self._ctrl_pair = pair
            self._ctrl_start = now
            return ('control_hold',)

        if now - self._ctrl_start < 0.5:
            return ('control_hold',)

        # Reset to prevent repeat triggers
        self._ctrl_pair = None
        self._ctrl_start = 0

        if pair == self.CTRL_SEPARATOR:
            if now - self._last_separator_time < 1.5:
                self._ctrl_pair = None; self._ctrl_start = 0
                return None
            self._last_separator_time = now
            self.inject_separator()
            self.last_cooldown_end = now + self.cooldown
            return ('separator',)

        if pair == self.CTRL_UNDO_WORD and len(self.buffer) > 0:
            removed = self.undo_last_word()
            self.last_cooldown_end = now + self.cooldown
            buf = [g for g, _ in self.buffer]
            return ('undo_word', removed, buf)

        if pair == self.CTRL_UNDO_SENTENCE and len(self.buffer) > 0:
            self.undo_sentence()
            self.last_cooldown_end = now + self.cooldown
            return ('undo_sentence',)

        return None

    def undo_last_word(self):
        if self.buffer:
            removed = self.buffer.pop()
            self.last_gesture = self.buffer[-1][0] if self.buffer else None
            return removed[0]
        return None

    def undo_sentence(self):
        self.buffer = []
        self.last_gesture = None
        self._pending_gesture = None

    def inject_separator(self):
        self.last_gesture = None
        self._pending_gesture = None
        # Deduplicate: sustained separator hold should only produce one "，"
        if self.buffer and self.buffer[-1][0] == '，':
            return
        self.buffer.append(('，', time.time()))

    # ================================================================
    # Transition detection
    # ================================================================

    def _dual_state(self, all_gestures):
        """Detect both_fist / both_open with tolerance for near-matches.
        Open_Palm sometimes recognized as Four (4 fingers), Closed_Fist as Three."""
        if not all_gestures or len(all_gestures) < 2:
            return 'other'
        g0, g1 = all_gestures[0], all_gestures[1]
        if g0 in ('Closed_Fist', 'Three') and g1 in ('Closed_Fist', 'Three'):
            return 'both_fist'
        if g0 in ('Open_Palm', 'Four') and g1 in ('Open_Palm', 'Four'):
            return 'both_open'
        return 'other'

    def _is_transition(self, kin_feat):
        if kin_feat is None:
            return False
        coord = float(kin_feat.get('coordination_change',
                                   __import__('numpy').zeros(10)).mean())
        speed = float(kin_feat.get('palm_speed',
                                   __import__('numpy').array([0.0]))[0])
        if coord > self.COORD_CHANGE_THRESHOLD and \
           speed > self.PALM_SPEED_THRESHOLD:
            return True
        if time.time() - self.last_gesture_change_time < self.TRANSITION_DEAD_ZONE:
            return True
        return False

    # ================================================================
    # State transitions
    # ================================================================

    def _start_recording(self):
        self.state = 'RECORDING'
        self.buffer = []
        self._last_buffer_mode = None
        self.last_gesture_time = time.time()
        self.last_cooldown_end = time.time() + self.cooldown
        self.last_gesture = None
        self.last_gesture_change_time = time.time()
        self._pending_gesture = None
        self._pending_gesture_start = 0
        self._recording_fist_start = 0
        self._single_fist_start = 0
        self._ctrl_pair = None
        self._ctrl_start = 0
        self._prev_dual_state = 'both_open'
        self._current_mode = None          # fresh recording → no persistent mode
        self._current_mode_name = '基础'
        self._last_start_time = time.time()
        return ('start',)

    def _end_recording(self):
        self.state = 'IDLE'
        if len(self.buffer) < 1:
            self.buffer = []
            self._reset_end_state()
            return None
        gestures = [g for g, _ in self.buffer]
        self._last_raw_tokens = gestures[:]
        self.buffer = []
        result_text = self._parse_sequence(gestures)
        self.sentence_history.append({
            'text': result_text, 'gestures': gestures,
            'mode': self._current_mode_name, 'timestamp': time.time()
        })
        self._reset_end_state()
        return ('output', result_text, gestures)

    def _reset_end_state(self):
        """Reset tracking state so the next recording starts clean.
        _prev_dual_state is deliberately preserved — it carries the
        'both_fist' state from the end transition into the next IDLE,
        enabling the fist→open transition-based start to fire immediately.
        """
        self._pending_gesture = None
        self._pending_gesture_start = 0
        self._recording_fist_start = 0
        self._single_fist_start = 0
        self._idle_open_start = 0
        self._last_buffer_mode = None
        self._ctrl_pair = None
        self._ctrl_start = 0

    # ================================================================
    # Semantic parsing + DeepSeek
    # ================================================================

    def _parse_sequence(self, gestures, mode='base'):
        clean = [g for g in gestures if g and g != 'None' and g != '__SEP__']
        if not clean:
            return ''

        self.semantic_parser.flush()
        matched = None
        for gesture in clean:
            result = self.semantic_parser.add_token(gesture)
            if result:
                matched = result

        category = 'common'
        base_text = ''
        if matched:
            base_text = matched.get('chain', ' '.join(clean))
            category = matched.get('category', 'common')
        else:
            base_text = ' '.join(clean)

        if self.deepseek and self.deepseek.is_available():
            try:
                context = [h['text'] for h in self.sentence_history[-3:]]
                assembled = self.deepseek.reorder_and_assemble(
                    clean, category=category, mode=self._current_mode_name,
                    context=context
                )
                if assembled:
                    return assembled
                polished = self.deepseek.hospital_polish(base_text)
                if polished:
                    return polished
            except Exception as e:
                print(f'[DeepSeek] error (using raw): {e}')

        return base_text

    # ================================================================
    # Utilities
    # ================================================================

    def cancel(self):
        self.state = 'IDLE'
        self.buffer = []
        self.last_gesture = None
        self._pending_gesture = None
        self._prev_dual_state = None
        self._recording_fist_start = 0
        self._single_fist_start = 0
        self._idle_open_start = 0
        self._ctrl_pair = None
        self._ctrl_start = 0
        self.semantic_parser.flush()

    def force_end(self):
        """Public API: force-end current recording. Returns action tuple or None."""
        if self.state != 'RECORDING' or len(self.buffer) < 1:
            self.state = 'IDLE'
            self.buffer = []
            return None
        return self._end_recording()

    def get_history(self, limit=50):
        return self.sentence_history[-limit:]

    def get_state(self):
        return self.state

    def get_mode(self):
        return self._current_mode_name
