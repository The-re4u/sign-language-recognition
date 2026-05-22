# coding:utf-8
"""Unit tests for SentenceRecorder state machine — core gesture recording logic."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.semantic.temporal_parser import TemporalSemanticParser
from core.semantic.sentence_recorder import SentenceRecorder


class TestSentenceRecorder:
    """Test the core state machine: IDLE → RECORDING → output."""

    def setup_method(self):
        self.parser = TemporalSemanticParser('config/hospital_chains.json')
        self.sr = SentenceRecorder(self.parser, cooldown=0.05,
                                   min_gesture_duration=0.05)

    # ---- State transitions ----

    def test_initial_state_is_idle(self):
        assert self.sr.get_state() == 'IDLE'

    def test_start_recording_both_open_sustained(self):
        # First frame: timer starts
        r = self.sr.process('Open_Palm', 2, ['Open_Palm', 'Open_Palm'])
        assert r is None
        time.sleep(0.6)
        # Second frame: sustained > 0.5s → start
        r = self.sr.process('Open_Palm', 2, ['Open_Palm', 'Open_Palm'])
        assert r == ('start',)
        assert self.sr.get_state() == 'RECORDING'

    def test_start_recording_fist_to_open_transition(self):
        # Simulate fist→open transition
        self.sr._prev_dual_state = 'both_fist'
        r = self.sr.process('Open_Palm', 2, ['Open_Palm', 'Open_Palm'])
        assert r == ('start',)
        assert self.sr.get_state() == 'RECORDING'

    def test_end_recording_both_fist_sustained(self):
        self._start()
        time.sleep(0.1)
        r = self.sr.process('Closed_Fist', 2, ['Closed_Fist', 'Closed_Fist'])
        time.sleep(1.6)
        r = self.sr.process('Closed_Fist', 2, ['Closed_Fist', 'Closed_Fist'])
        assert r is not None
        assert r[0] == 'output'
        assert self.sr.get_state() == 'IDLE'

    # ---- Content gesture recognition ----

    def test_number_mode_gesture(self):
        self._start()
        time.sleep(0.1)
        r = self.sr.process('Three', 2, ['Good', 'Three'])
        time.sleep(0.1)
        r = self.sr.process('Three', 2, ['Good', 'Three'])
        assert r is not None
        assert r[0] == 'buffer'
        assert '3' in r[1]

    def test_symptom_mode_gesture(self):
        self._start()
        time.sleep(0.1)
        r = self.sr.process('Closed_Fist', 2, ['Eight', 'Closed_Fist'])
        time.sleep(0.1)
        r = self.sr.process('Closed_Fist', 2, ['Eight', 'Closed_Fist'])
        assert r is not None
        assert r[0] == 'buffer'

    def test_single_hand_phrase(self):
        self._start()
        time.sleep(0.1)
        r = self.sr.process('Closed_Fist', 1, ['Closed_Fist'])
        time.sleep(0.1)
        r = self.sr.process('Closed_Fist', 1, ['Closed_Fist'])
        assert r is not None
        assert r[0] == 'buffer'

    def test_single_hand_yes(self):
        self._start()
        time.sleep(0.1)
        r = self.sr.process('Six', 1, ['Six'])
        time.sleep(0.1)
        r = self.sr.process('Six', 1, ['Six'])
        assert r is not None

    # ---- Control gestures ----

    def test_undo_word(self):
        self._start()
        time.sleep(0.1)
        self._add_gesture('Three', ['Good', 'Three'])
        self._add_gesture('Open_Palm', ['Good', 'Open_Palm'])
        # Trigger undo: 双手 Victory
        r = self.sr.process('Victory', 2, ['Victory', 'Victory'])
        assert r[0] == 'control_hold'
        time.sleep(0.6)
        r = self.sr.process('Victory', 2, ['Victory', 'Victory'])
        assert r[0] == 'undo_word'

    def test_separator_adds_comma(self):
        self._start()
        time.sleep(0.1)
        self._add_gesture('Closed_Fist', ['Eight', 'Closed_Fist'])  # 疼
        # Trigger separator: 双手 One
        r = self.sr.process('One', 2, ['One', 'One'])
        time.sleep(0.6)
        r = self.sr.process('One', 2, ['One', 'One'])
        assert r[0] == 'separator'

    def test_undo_sentence(self):
        self._start()
        time.sleep(0.1)
        self._add_gesture('Closed_Fist', ['Eight', 'Closed_Fist'])
        # Trigger undo sentence: 双手 Seven
        r = self.sr.process('Seven', 2, ['Seven', 'Seven'])
        time.sleep(0.6)
        r = self.sr.process('Seven', 2, ['Seven', 'Seven'])
        assert r[0] == 'undo_sentence'
        assert len(self.sr.buffer) == 0

    # ---- Mode persistence & timeout ----

    def test_persistent_mode_after_two_hand(self):
        self._start()
        time.sleep(0.1)
        self._add_gesture('Three', ['Good', 'Three'])
        # Now single hand should use persistent mode
        r = self.sr._classify_hands(['Three'])
        assert r[0] == 'Good'  # persistent mode
        assert r[1] == 'Three'

    def test_mode_reset_on_new_recording(self):
        self._start()
        time.sleep(0.1)
        self._add_gesture('Three', ['Good', 'Three'])
        assert self.sr._current_mode == 'Good'
        self.sr._end_recording()
        self.sr._start_recording()
        assert self.sr._current_mode is None  # reset on new recording

    def test_mode_control_gesture_not_leaked(self):
        self._start()
        time.sleep(0.1)
        r = self.sr.process('Victory', 2, ['Victory', 'Victory'])
        # During control hold, mode gesture should NOT enter buffer
        assert r[0] == 'control_hold'

    # ---- State reset ----

    def test_cancel_resets_state(self):
        self._start()
        time.sleep(0.1)
        self._add_gesture('Three', ['Good', 'Three'])
        self.sr.cancel()
        assert self.sr.get_state() == 'IDLE'
        assert len(self.sr.buffer) == 0

    def test_force_end_produces_output(self):
        self._start()
        time.sleep(0.1)
        self._add_gesture('Closed_Fist', ['Eight', 'Closed_Fist'])
        r = self.sr.force_end()
        assert r is not None
        assert r[0] == 'output'
        assert self.sr.get_state() == 'IDLE'

    # ---- Helpers ----

    def _start(self):
        time.sleep(0.6)
        return self.sr.process('Open_Palm', 2, ['Open_Palm', 'Open_Palm'])

    def _add_gesture(self, content, all_gestures):
        time.sleep(0.1)
        self.sr.process(content, len(all_gestures), all_gestures)
        time.sleep(0.1)
        return self.sr.process(content, len(all_gestures), all_gestures)
