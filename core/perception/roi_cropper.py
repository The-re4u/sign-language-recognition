# coding:utf-8
"""OpenCV-based hand ROI cropping using affine transformation."""
import cv2
import numpy as np


class ROICropper:
    """Crops and normalizes hand region from a frame using 21-landmark bounding box."""

    def __init__(self, target_size=(96, 96), margin=0.2):
        self.target_size = target_size
        self.margin = margin
        self._last_bbox = None

    def compute_bbox(self, hand_landmarks_wrapper, image_shape):
        """Return (xmin, ymin, xmax, ymax) from 21 landmarks."""
        h, w = image_shape[:2]
        xs, ys = [], []
        for lm in hand_landmarks_wrapper.landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return (xmin, ymin, xmax, ymax)

    def crop(self, image, hand_landmarks_wrapper):
        """Crop and normalize hand ROI to target_size."""
        h, w = image.shape[:2]
        bbox = self.compute_bbox(hand_landmarks_wrapper, image.shape)
        xmin, ymin, xmax, ymax = bbox

        bw = xmax - xmin
        bh = ymax - ymin
        xmin = max(0, int(xmin - bw * self.margin))
        ymin = max(0, int(ymin - bh * self.margin))
        xmax = min(w, int(xmax + bw * self.margin))
        ymax = min(h, int(ymax + bh * self.margin))

        if xmax <= xmin or ymax <= ymin:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)

        roi = image[ymin:ymax, xmin:xmax]
        roi = cv2.resize(roi, self.target_size)
        return roi
