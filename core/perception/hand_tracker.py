# coding:utf-8
"""MediaPipe HandLandmarker wrapper with GPU delegate support."""
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


class HandLandmarksWrapper:
    """Wrapper for mp.tasks landmark list → drawing_utils / tools compatibility."""
    def __init__(self, landmarks):
        self.landmark = landmarks


def _create_base_options(model_path, use_gpu=True):
    """Create BaseOptions, trying GPU first with CPU fallback."""
    if use_gpu:
        try:
            delegate = mp_python.BaseOptions.Delegate.GPU
            opts = mp_python.BaseOptions(model_asset_path=model_path, delegate=delegate)
            # Quick validation: try creating a detector to see if GPU works
            test_options = mp_vision.HandLandmarkerOptions(
                base_options=opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2)
            mp_vision.HandLandmarker.create_from_options(test_options).close()
            print('[HandTracker] GPU delegate enabled (NVIDIA GTX 1660 Ti)')
            return opts
        except Exception as e:
            print(f'[HandTracker] GPU delegate failed ({e}), falling back to CPU')

    print('[HandTracker] Using CPU delegate')
    return mp_python.BaseOptions(model_asset_path=model_path)


class HandTracker:
    """Wraps MediaPipe HandLandmarker (.task) for IMAGE and VIDEO running modes.

    Tries GPU delegate first (Windows + NVIDIA), falls back to CPU automatically.
    """

    def __init__(self, model_path='hand_landmarker.task', num_hands=2,
                 min_detection_confidence=0.6, min_presence_confidence=0.6,
                 min_tracking_confidence=0.6, use_gpu=True):
        base_options = _create_base_options(model_path, use_gpu)

        img_options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self._img_detector = mp_vision.HandLandmarker.create_from_options(img_options)

        vid_options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self._vid_detector = mp_vision.HandLandmarker.create_from_options(vid_options)

        self.drawing_styles = mp.solutions.drawing_styles
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS

    def detect_image(self, rgb_image):
        """Detect hands in a single image (IMAGE mode)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self._img_detector.detect(mp_image)

    def detect_video(self, rgb_image, timestamp_ms):
        """Detect hands in a video frame (VIDEO mode)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self._vid_detector.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self._img_detector.close()
        self._vid_detector.close()
