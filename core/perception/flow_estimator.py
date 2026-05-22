# coding:utf-8
"""Dense optical flow estimation using OpenCV Farneback."""
import cv2
import numpy as np


class FlowEstimator:
    """Computes dense optical flow between consecutive frames."""

    def __init__(self):
        self.prev_gray = None

    def compute(self, current_gray):
        """Return flow field (h, w, 2) between previous and current frame."""
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return np.zeros((*current_gray.shape, 2), dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, current_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = current_gray
        return flow

    def reset(self):
        self.prev_gray = None
