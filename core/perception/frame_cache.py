# coding:utf-8
"""Sliding window frame cache for temporal modeling."""
from collections import deque


class FrameCache:
    """Sliding window cache storing up to window_size FrameData objects."""

    def __init__(self, window_size=32):
        self.window = deque(maxlen=window_size)

    def push(self, frame_data):
        self.window.append(frame_data)

    def is_full(self):
        return len(self.window) == self.window.maxlen

    def is_ready(self, min_frames=16):
        return len(self.window) >= min_frames

    def get_window(self):
        return list(self.window)

    def get_latest(self):
        return self.window[-1] if self.window else None

    def get_timestamps(self):
        return [f.timestamp for f in self.window]

    def clear(self):
        self.window.clear()

    def __len__(self):
        return len(self.window)
