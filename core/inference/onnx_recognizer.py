# coding:utf-8
"""ONNX Runtime inference wrapper for the sign language recognition model.

Loads the exported ONNX model and provides a simple predict() interface
that accepts a sequence of per-frame fused features and returns class logits.

Designed to slot into the real-time pipeline alongside RuleRecognizer.
"""
import numpy as np


class ONNXRecognizer:
    """Lightweight ONNX inference wrapper for the SlowFast TCN model.

    Input:  fused feature sequence [T, 256]  (from SpatialGCN + MotionEncoder + Fusion)
    Output: frame-wise class logits [T, num_classes]

    Falls back gracefully if ONNX model or onnxruntime is unavailable.
    """

    def __init__(self, model_path='models/sign_recognizer.onnx', num_classes=14):
        self.model_path = model_path
        self.num_classes = num_classes
        self.session = None
        self._available = False
        self._init_session()

    def _init_session(self):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self._available = True
            print(f'[ONNXRecognizer] Loaded: {self.model_path}')
        except FileNotFoundError:
            print(f'[ONNXRecognizer] Model not found: {self.model_path} '
                  '(train + export first)')
        except ImportError:
            print('[ONNXRecognizer] onnxruntime not installed, ONNX inference disabled')
        except Exception as e:
            print(f'[ONNXRecognizer] Failed to load: {e}')

    def is_available(self):
        return self._available and self.session is not None

    def predict(self, features, return_probs=True):
        """Run inference on a feature sequence.

        Args:
            features: [T, 256] numpy array of fused per-frame features
            return_probs: if True, apply softmax and return probabilities

        Returns:
            (class_ids [T], confidence [T]) if return_probs=True
            else logits [T, num_classes]
        """
        if not self.is_available():
            return None

        # Add batch dimension: [1, T, 256]
        inp = features[np.newaxis, :, :].astype(np.float32)
        outputs = self.session.run(None, {'input': inp})
        logits = outputs[0][0]  # [T, num_classes]

        if return_probs:
            probs = self._softmax(logits)
            class_ids = np.argmax(probs, axis=1)
            confidence = np.max(probs, axis=1)
            return class_ids, confidence
        return logits

    def predict_sequence_label(self, features):
        """Predict the overall sequence label (majority vote over frames).

        Args:
            features: [T, 256] fused features

        Returns:
            (label_id, confidence) or None
        """
        result = self.predict(features, return_probs=True)
        if result is None:
            return None
        class_ids, confidence = result
        # Majority vote over non-low-confidence frames
        confident_frames = class_ids[confidence > 0.3]
        if len(confident_frames) == 0:
            confident_frames = class_ids
        from collections import Counter
        most_common = Counter(confident_frames.tolist()).most_common(1)[0]
        return most_common[0], float(most_common[1]) / len(confident_frames)

    @staticmethod
    def _softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)
