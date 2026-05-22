# coding:utf-8
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HandInfo:
    """Per-hand detection data for a single frame."""
    label: str  # 'Right' or 'Left'
    landmarks: list  # 21 NormalizedLandmark objects
    bbox: tuple  # (xmin, ymin, xmax, ymax)
    roi_image: np.ndarray = None  # Cropped hand ROI (96x96 RGB)
    finger_up: list = field(default_factory=list)  # 5 bools
    finger_count: int = 0
    gesture: str = 'None'


@dataclass
class FrameData:
    """Single frame stored in the sliding window cache."""
    timestamp: float
    image_rgb: np.ndarray = None
    image_gray: np.ndarray = None
    hands: list = field(default_factory=list)  # List[HandInfo]
    optical_flow: np.ndarray = None
