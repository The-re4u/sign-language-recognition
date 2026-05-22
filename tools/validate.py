# coding:utf-8
"""
Clinical-grade validation suite for sign language recognition system v3.0.

Methodology aligned with:
  Wagh et al. (2025) "Using MediaPipe to track upper-limb reaching movements
  after stroke." J NeuroEngineering Rehabil, 22, 268.

Validates:
  1. Per-gesture precision/recall/F1 + confusion matrix
  2. PIP-sensitive vs PIP-robust gesture performance comparison
  3. Boundary condition tests (speed, noise, occlusion, lighting, distance, resolution)
  4. Ablation study (--ablate: PIP compensation, kinematic features)
  5. System performance benchmarking (--benchmark)
  6. Bootstrap confidence intervals + confidence calibration (ECE)

Usage:
  python tools/validate.py --data data/annotations_all.json --data_dir data/sequences
  python tools/validate.py --data data/annotations_all.json --speed_test --noise_test --benchmark --ablate
  python tools/validate.py --data data/annotations.json --report data/validation_report.md
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Validate gesture recognition system')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to annotation JSON')
    parser.add_argument('--data_dir', type=str, default='data/sequences',
                        help='Directory for .npy sequence files')
    parser.add_argument('--report', type=str, default=None,
                        help='Output markdown report path')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Bootstrap iterations for confidence intervals')
    parser.add_argument('--speed_test', action='store_true',
                        help='Run speed degradation test')
    parser.add_argument('--noise_test', action='store_true',
                        help='Run noise robustness test')
    parser.add_argument('--occlusion_test', action='store_true',
                        help='Run occlusion robustness test (0-50% finger occlusion)')
    parser.add_argument('--lighting_test', action='store_true',
                        help='Run lighting variability test (simulated brightness)')
    parser.add_argument('--distance_test', action='store_true',
                        help='Run camera distance variability test (simulated scale)')
    parser.add_argument('--resolution_test', action='store_true',
                        help='Run resolution degradation test (subsampled keypoints)')
    parser.add_argument('--ablate', action='store_true',
                        help='Run ablation study (PIP compensation, kinematic features)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run system performance benchmarking')
    parser.add_argument('--benchmark_iters', type=int, default=100,
                        help='Iterations for performance benchmarking')
    return parser.parse_args()


# ============================================================
# PIP-sensitive gestures (from literature)
# ============================================================
PIP_SENSITIVE = {'Victory', 'Six', 'Three', 'Four', 'Seven', 'Nine'}


def load_data(annotation_path, data_dir):
    """Load annotated sequences."""
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    samples = []
    for ann in annotations:
        # frames_path may be relative to project root or to data_dir
        seq_path = os.path.join(data_dir, os.path.basename(ann.get('frames_path', '')))
        if not os.path.exists(seq_path):
            seq_path = ann.get('frames_path', '')  # try as-is
        if os.path.exists(seq_path):
            kp_seq = np.load(seq_path)
            samples.append({
                'video_id': ann['video_id'],
                'label': ann['label'],
                'keypoints': kp_seq,
            })

    print(f'Loaded {len(samples)} sequences ({len(annotations)} annotated)')
    return samples


def recognize_sequence(kp_seq, recognizer):
    """Run rule-based recognition on a keypoint sequence.
    Returns list of (gesture, confidence) per frame."""
    results = []
    for frame_idx in range(kp_seq.shape[0]):
        kp_frame = kp_seq[frame_idx]  # [21, 3]

        # Wrap as HandLandmarksWrapper-compatible object
        class FakeLandmark:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class FakeWrapper:
            def __init__(self, kp):
                self.landmark = [FakeLandmark(float(k[0]), float(k[1]), float(k[2]))
                                 for k in kp]

        wrapper = FakeWrapper(kp_frame)
        finger_up, finger_count, gesture, confidence = \
            recognizer.recognize_with_confidence(wrapper)
        results.append((gesture, confidence))

    return results


def majority_vote(gestures, min_ratio=0.3):
    """Return majority gesture from frame-level predictions."""
    from collections import Counter
    counts = Counter(gestures)
    total = len(gestures)
    most_common, count = counts.most_common(1)[0]
    if count / total >= min_ratio:
        return most_common
    return most_common  # fallback even below threshold


def compute_metrics(y_true, y_pred):
    """Compute per-class precision, recall, F1."""
    all_labels = sorted(set(list(y_true) + list(y_pred)))
    metrics = {}

    for label in all_labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for t in y_true if t == label)
        }

    # Macro average
    metrics['macro_avg'] = {
        'precision': np.mean([m['precision'] for m in metrics.values()]),
        'recall': np.mean([m['recall'] for m in metrics.values()]),
        'f1': np.mean([m['f1'] for m in metrics.values()]),
        'support': len(y_true)
    }

    # Accuracy
    metrics['accuracy'] = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    return metrics


def bootstrap_ci(data, metric_fn, n_bootstrap=1000, alpha=0.05):
    """Bootstrap confidence interval for a metric."""
    estimates = []
    n = len(data)
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = [data[rng.randint(0, n)] for _ in range(n)]
        estimates.append(metric_fn(sample))
    estimates = np.array(estimates)
    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))
    mean = np.mean(estimates)
    return mean, lower, upper


def test_speed_degradation(recognizer, samples, speed_levels=None):
    """v2.1: Test recognition accuracy under different motion speeds.
    Reference: Sprague et al. (2025) found MediaPipe degradation at high speed.
    """
    if speed_levels is None:
        speed_levels = {
            'slow': 1.0,     # original speed
            'moderate': 2.0,  # 2x speed (subsample)
            'fast': 4.0,      # 4x speed (heavy subsample)
        }

    results = {}
    for speed_name, speed_factor in speed_levels.items():
        y_true, y_pred = [], []

        for sample in samples:
            kp_seq = sample['keypoints']
            # Subsample to simulate higher speed (fewer frames = faster motion)
            indices = np.arange(0, len(kp_seq), speed_factor).astype(int)
            if len(indices) < 2:
                continue
            fast_seq = kp_seq[indices]

            frame_results = recognize_sequence(fast_seq, recognizer)
            gestures = [g for g, _ in frame_results]
            pred = majority_vote(gestures)
            y_true.append(sample['label'])
            y_pred.append(pred)

        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        results[speed_name] = {'accuracy': acc, 'n_samples': len(y_true)}

    return results


def test_noise_robustness(recognizer, samples, noise_levels=None):
    """v2.1: Test recognition accuracy under increasing keypoint noise.
    Reference: Maggioni et al. (2025) quantified MediaPipe jitter range.
    """
    if noise_levels is None:
        noise_levels = {
            'clean': 0.0,
            'mild_jitter': 0.005,   # Typical MediaPipe jitter
            'moderate_jitter': 0.015,  # Above-average jitter
            'severe_jitter': 0.030,   # Worst-case single-camera jitter
        }

    results = {}
    for noise_name, noise_sigma in noise_levels.items():
        y_true, y_pred = [], []
        rng = np.random.RandomState(42)

        for sample in samples:
            kp_seq = sample['keypoints'].copy()
            if noise_sigma > 0:
                noise = rng.randn(*kp_seq.shape).astype(np.float32) * noise_sigma
                kp_seq[:, :, :2] += noise[:, :, :2]
                kp_seq[:, :, :2] = np.clip(kp_seq[:, :, :2], 0.0, 1.0)

            frame_results = recognize_sequence(kp_seq, recognizer)
            gestures = [g for g, _ in frame_results]
            pred = majority_vote(gestures)
            y_true.append(sample['label'])
            y_pred.append(pred)

        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        results[noise_name] = {'accuracy': acc, 'n_samples': len(y_true)}

    return results


def test_confidence_calibration(y_true, y_pred, confidences):
    """v2.1: Check if confidence scores predict correctness.
    Binning confidence scores and computing accuracy per bin.
    """
    bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
    calibration = []

    for low, high in bins:
        bin_indices = [i for i, c in enumerate(confidences) if low <= c < high]
        if not bin_indices:
            continue
        bin_correct = sum(1 for i in bin_indices if y_true[i] == y_pred[i])
        bin_acc = bin_correct / len(bin_indices)
        calibration.append({
            'bin': f'{low:.2f}-{high:.2f}',
            'accuracy': bin_acc,
            'count': len(bin_indices),
            'expected_confidence': (low + high) / 2
        })

    # Expected Calibration Error (ECE)
    ece = 0.0
    total = len(y_true)
    for cal in calibration:
        diff = abs(cal['accuracy'] - cal['expected_confidence'])
        ece += diff * cal['count'] / total

    return calibration, ece


def plot_reliability_diagram(calibration, ece, output_path='docs/reliability_diagram.png'):
    """Generate reliability diagram (calibration curve) for thesis.

    X-axis: expected confidence (binned)
    Y-axis: observed accuracy
    Diagonal = perfect calibration
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[Plot] matplotlib not installed, skipping reliability diagram')
        return

    bins = [c['bin'] for c in calibration]
    accs = [c['accuracy'] for c in calibration]
    expected = [c['expected_confidence'] for c in calibration]
    counts = [c['count'] for c in calibration]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    ax.bar(bins, accs, width=0.08, alpha=0.7, color='#2196F3', label='Observed')
    ax.scatter(expected, accs, s=[c * 2 for c in counts], c='#F44336', zorder=5)

    for i, (b, a, c) in enumerate(zip(bins, accs, counts)):
        ax.annotate(f'n={c}', (expected[i], accs[i] + 0.02),
                    ha='center', fontsize=8, color='#666')

    ax.set_xlabel('Expected Confidence', fontsize=11)
    ax.set_ylabel('Observed Accuracy', fontsize=11)
    ax.set_title(f'Reliability Diagram (ECE = {ece:.4f})', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot] Reliability diagram saved: {output_path}')


def compute_confusion_matrix(y_true, y_pred):
    """Build normalized confusion matrix. Returns (labels, matrix, top_errors)."""
    labels = sorted(set(list(y_true) + list(y_pred)))
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    # Find top off-diagonal confusion pairs
    errors = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                errors.append((labels[i], labels[j], int(cm[i, j])))
    errors.sort(key=lambda x: -x[2])
    return labels, cm, errors[:15]


def format_confusion_matrix(labels, cm):
    """Render text-based confusion matrix for markdown report."""
    n = len(labels)
    max_label = max(len(l) for l in labels)
    col_w = 6
    lines = []
    header = ' ' * (max_label + 2) + ''.join(f'{l:>{col_w}}' for l in labels)
    lines.append(f'```')
    lines.append(header)
    for i, label in enumerate(labels):
        row = f'{label:>{max_label}}  '
        row += ''.join(f'{cm[i,j]:{col_w}d}' for j in range(n))
        lines.append(row)
    lines.append(f'```')
    return '\n'.join(lines)


def test_occlusion_robustness(recognizer, samples, occlusion_levels=None):
    """Test accuracy under simulated finger occlusion."""
    if occlusion_levels is None:
        occlusion_levels = {'clean': 0, '10%': 2, '25%': 5, '50%': 10}

    rng = np.random.RandomState(42)
    results = {}
    all_landmark_indices = list(range(21))

    for occ_name, n_hidden in occlusion_levels.items():
        y_true, y_pred = [], []
        for sample in samples:
            kp_seq = sample['keypoints'].copy()
            if n_hidden > 0:
                hidden = set(rng.choice(all_landmark_indices, n_hidden, replace=False))
                for frame_idx in range(kp_seq.shape[0]):
                    for h in hidden:
                        kp_seq[frame_idx, h, :2] = 0.0  # zero out hidden landmarks

            frame_results = recognize_sequence(kp_seq, recognizer)
            gestures = [g for g, _ in frame_results]
            pred = majority_vote(gestures)
            y_true.append(sample['label'])
            y_pred.append(pred)

        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        results[occ_name] = {'accuracy': acc, 'occluded_landmarks': n_hidden, 'n_samples': len(y_true)}

    return results


def test_lighting_variability(recognizer, samples, brightness_levels=None):
    """Test accuracy under simulated lighting changes (keypoint coordinate jitter at scale)."""
    if brightness_levels is None:
        brightness_levels = {'bright': 0.002, 'normal': 0.006, 'dim': 0.015, 'dark': 0.030}

    rng = np.random.RandomState(42)
    results = {}
    for light_name, jitter_sigma in brightness_levels.items():
        y_true, y_pred = [], []
        for sample in samples:
            kp_seq = sample['keypoints'].copy()
            if jitter_sigma > 0:
                noise = rng.randn(*kp_seq.shape).astype(np.float32) * jitter_sigma
                kp_seq[:, :, :2] += noise[:, :, :2]
                kp_seq[:, :, :2] = np.clip(kp_seq[:, :, :2], 0.0, 1.0)

            frame_results = recognize_sequence(kp_seq, recognizer)
            gestures = [g for g, _ in frame_results]
            pred = majority_vote(gestures)
            y_true.append(sample['label'])
            y_pred.append(pred)

        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        results[light_name] = {'accuracy': acc, 'jitter_std': jitter_sigma, 'n_samples': len(y_true)}

    return results


def test_distance_variability(recognizer, samples, scale_levels=None):
    """Test accuracy under simulated camera distance (scale keypoint coordinates)."""
    if scale_levels is None:
        scale_levels = {'near (30cm)': 1.3, 'normal (60cm)': 1.0, 'far (90cm)': 0.7, 'very_far (120cm)': 0.5}

    results = {}
    for dist_name, scale in scale_levels.items():
        y_true, y_pred = [], []
        for sample in samples:
            kp_seq = sample['keypoints'].copy()
            centroid = kp_seq[:, :, :2].mean(axis=1, keepdims=True)
            kp_seq[:, :, :2] = (kp_seq[:, :, :2] - centroid) * scale + centroid
            kp_seq[:, :, :2] = np.clip(kp_seq[:, :, :2], 0.0, 1.0)

            frame_results = recognize_sequence(kp_seq, recognizer)
            gestures = [g for g, _ in frame_results]
            pred = majority_vote(gestures)
            y_true.append(sample['label'])
            y_pred.append(pred)

        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        results[dist_name] = {'accuracy': acc, 'scale': scale, 'n_samples': len(y_true)}

    return results


def test_resolution_degradation(recognizer, samples, res_levels=None):
    """Test accuracy under simulated lower camera resolution (quantized keypoints)."""
    if res_levels is None:
        res_levels = {'1080p': 0, '720p': 2, '480p': 4, '240p': 6}

    results = {}
    for res_name, quant_bits in res_levels.items():
        y_true, y_pred = [], []
        for sample in samples:
            kp_seq = sample['keypoints'].copy()
            if quant_bits > 0:
                levels = 2 ** (10 - quant_bits)
                kp_seq[:, :, :2] = np.round(kp_seq[:, :, :2] * levels) / levels

            frame_results = recognize_sequence(kp_seq, recognizer)
            gestures = [g for g, _ in frame_results]
            pred = majority_vote(gestures)
            y_true.append(sample['label'])
            y_pred.append(pred)

        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        results[res_name] = {'accuracy': acc, 'quant_bits': quant_bits, 'n_samples': len(y_true)}

    return results


def run_ablation_study(recognizer, samples, ablate_pip=True, ablate_kin=True):
    """Ablation study: measure impact of PIP compensation.
    Monkeys-patches the recognizer to disable features.

    Note: kinematic feature ablation only meaningful in full pipeline (backend/server.py).
    Rule-based path does not use kinematic features — this study focuses on PIP compensation.
    """
    import core.fallback.rule_recognizer as rrmod

    configs = {
        'full_system': True,
        'no_pip_compensation': False,
    }

    results = {}
    for config_name, pip_enabled in configs.items():
        # Monkey-patch to disable PIP check
        original_set = rrmod.PIP_SENSITIVE_GESTURES
        if not pip_enabled:
            rrmod.PIP_SENSITIVE_GESTURES = set()
        else:
            rrmod.PIP_SENSITIVE_GESTURES = original_set

        y_true, y_pred = [], []
        for sample in samples:
            kp_seq = sample['keypoints']
            frame_results = recognize_sequence(kp_seq, recognizer)
            gestures = [g for g, _ in frame_results]
            pred = majority_vote(gestures)
            y_true.append(sample['label'])
            y_pred.append(pred)

        metrics = compute_metrics(y_true, y_pred)
        results[config_name] = {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_avg']['f1'],
            'pip_compensation': pip_enabled,
            'kin_features': True,
        }

        # Restore original
        rrmod.PIP_SENSITIVE_GESTURES = original_set

    return results


def run_benchmark(recognizer, samples, iters=100):
    """Performance benchmarking: measure per-inference latency statistics."""
    times_ms = []
    for _ in range(iters):
        idx = np.random.randint(0, len(samples))
        kp_seq = samples[idx]['keypoints']
        t0 = time.perf_counter()
        recognize_sequence(kp_seq, recognizer)
        elapsed = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed)

    times = np.array(times_ms)
    return {
        'mean_ms': float(np.mean(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'std_ms': float(np.std(times)),
        'iterations': iters,
        'fps_estimate': round(1000.0 / np.mean(times), 1),
    }


def generate_report(metrics, speed_results, noise_results, calibration, ece,
                    pip_comparison, ci_results, output_path,
                    confusion_data=None, occlusion_results=None,
                    lighting_results=None, distance_results=None,
                    resolution_results=None, ablation_results=None,
                    benchmark_results=None):
    """Generate a markdown validation report."""
    lines = []
    lines.append('# 手势识别系统 v3.0 — 临床级验证报告')
    lines.append('')
    lines.append(f'*自动生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}*')
    lines.append('')
    lines.append('> 方法论依据: Wagh et al. (2025) *J NeuroEngineering Rehabil*')
    lines.append('')

    # Overall metrics
    lines.append('## 一、整体性能指标')
    lines.append('')
    lines.append('| 指标 | 值 | 95% CI |')
    lines.append('|------|-----|--------|')
    lines.append(f'| Accuracy | {metrics["accuracy"]:.3f} | [{ci_results["accuracy"][0]:.3f}, {ci_results["accuracy"][1]:.3f}] |')
    lines.append(f'| Macro F1 | {metrics["macro_avg"]["f1"]:.3f} | [{ci_results["f1"][0]:.3f}, {ci_results["f1"][1]:.3f}] |')
    lines.append(f'| Macro Precision | {metrics["macro_avg"]["precision"]:.3f} | — |')
    lines.append(f'| Macro Recall | {metrics["macro_avg"]["recall"]:.3f} | — |')
    lines.append('')

    # Per-gesture metrics
    lines.append('## 二、分手势性能指标')
    lines.append('')
    lines.append('| 手势 | Precision | Recall | F1 | Support | PIP敏感 |')
    lines.append('|------|-----------|--------|----|---------|---------|')
    for label in sorted(metrics.keys()):
        if label in ('macro_avg', 'accuracy'):
            continue
        m = metrics[label]
        pip_tag = '[PIP] YES' if label in PIP_SENSITIVE else 'no'
        lines.append(f'| {label} | {m["precision"]:.3f} | {m["recall"]:.3f} | '
                     f'{m["f1"]:.3f} | {m["support"]} | {pip_tag} |')
    lines.append('')

    # PIP comparison
    lines.append('## 三、PIP 关节敏感度分析')
    lines.append('')
    lines.append('> 依据: Sprague et al. (2025) 与 Maggioni et al. (2025) 均报告 PIP 关节是 MediaPipe 误差最大的关节')
    lines.append('')
    lines.append('| 手势类型 | 平均 F1 | 手势数 |')
    lines.append('|----------|---------|--------|')
    if pip_comparison:
        for group, data in pip_comparison.items():
            lines.append(f'| {group} | {data["avg_f1"]:.3f} | {data["count"]} |')
    lines.append('')

    # Speed degradation
    if speed_results:
        lines.append('## 四、速度退化测试')
        lines.append('')
        lines.append('> 依据: Sprague et al. (2025) 发现高速运动时 MediaPipe RMSE 从 14.8° 恶化至 22.5°')
        lines.append('')
        lines.append('| 速度等级 | 准确率 | 样本数 |')
        lines.append('|----------|--------|--------|')
        baseline_acc = speed_results.get('slow', {}).get('accuracy', 0)
        for speed_name, result in speed_results.items():
            degradation = ''
            if baseline_acc > 0 and speed_name != 'slow':
                delta = (baseline_acc - result['accuracy']) * 100
                degradation = f' (-{delta:.1f}%)'
            lines.append(f'| {speed_name} | {result["accuracy"]:.3f}{degradation} | {result["n_samples"]} |')
        lines.append('')

    # Noise robustness
    if noise_results:
        lines.append('## 五、噪声鲁棒性测试')
        lines.append('')
        lines.append('> 依据: Hamaguchi et al. (2024) 报告 MediaPipe MAD 2.46° (原始) → 0.81° (平滑后)')
        lines.append('')
        lines.append('| 噪声等级 | σ | 准确率 | 样本数 |')
        lines.append('|----------|---|--------|--------|')
        for noise_name, result in noise_results.items():
            sigma = {'clean': 0.0, 'mild_jitter': 0.005,
                     'moderate_jitter': 0.015, 'severe_jitter': 0.030}.get(noise_name, 0)
            lines.append(f'| {noise_name} | {sigma:.3f} | {result["accuracy"]:.3f} | {result["n_samples"]} |')
        lines.append('')

    # Confidence calibration
    if calibration:
        lines.append('## 六、置信度校准')
        lines.append('')
        lines.append(f'**Expected Calibration Error (ECE): {ece:.4f}** (越低越好)')
        lines.append('')
        lines.append('| 置信度区间 | 实际准确率 | 期望置信度 | 样本数 |')
        lines.append('|------------|-----------|-----------|--------|')
        for cal in calibration:
            lines.append(f'| {cal["bin"]} | {cal["accuracy"]:.3f} | '
                         f'{cal["expected_confidence"]:.2f} | {cal["count"]} |')
        lines.append('')

    # Confusion matrix
    if confusion_data:
        labels, cm, top_errors = confusion_data
        lines.append('## 七、混淆矩阵')
        lines.append('')
        lines.append(format_confusion_matrix(labels, cm))
        lines.append('')
        if top_errors:
            lines.append('**Top 混淆对**：')
            for true_l, pred_l, count in top_errors[:10]:
                lines.append(f'- `{true_l}` → `{pred_l}` ({count} 次)')
        lines.append('')

    # Ablation study
    if ablation_results:
        lines.append('## 八、消融实验')
        lines.append('')
        lines.append('| 配置 | PIP补偿 | 运动学特征 | Accuracy | Macro F1 | Δ Acc |')
        lines.append('|------|---------|-----------|----------|---------|-------|')
        base_acc = ablation_results.get('full_system', {}).get('accuracy', 1.0)
        for config_name, data in ablation_results.items():
            pip_icon = '✅' if data['pip_compensation'] else '❌'
            kin_icon = '✅' if data['kin_features'] else '❌'
            delta = data['accuracy'] - base_acc
            delta_str = f'{-abs(delta):+.1%}' if delta < 0 else f'{delta:+.1%}'
            lines.append(f'| {config_name} | {pip_icon} | {kin_icon} | '
                         f'{data["accuracy"]:.3f} | {data["macro_f1"]:.3f} | {delta_str} |')
        lines.append('')

    # New robustness tests
    if occlusion_results:
        lines.append('## 九、遮挡鲁棒性测试')
        lines.append('')
        lines.append('| 遮挡程度 | 隐藏关键点 | Accuracy |')
        lines.append('|----------|-----------|----------|')
        for name, r in occlusion_results.items():
            lines.append(f'| {name} | {r["occluded_landmarks"]} | {r["accuracy"]:.3f} |')
        lines.append('')

    if lighting_results:
        lines.append('## 十、光照变化测试')
        lines.append('')
        lines.append('| 光照条件 | 噪声标准差 | Accuracy |')
        lines.append('|----------|----------|----------|')
        for name, r in lighting_results.items():
            lines.append(f'| {name} | {r["jitter_std"]:.4f} | {r["accuracy"]:.3f} |')
        lines.append('')

    if distance_results:
        lines.append('## 十一、摄像头距离测试')
        lines.append('')
        lines.append('| 距离 | 缩放系数 | Accuracy |')
        lines.append('|------|---------|----------|')
        for name, r in distance_results.items():
            lines.append(f'| {name} | {r["scale"]:.2f} | {r["accuracy"]:.3f} |')
        lines.append('')

    if resolution_results:
        lines.append('## 十二、分辨率退化测试')
        lines.append('')
        lines.append('| 模拟分辨率 | 量化位 | Accuracy |')
        lines.append('|-----------|--------|----------|')
        for name, r in resolution_results.items():
            lines.append(f'| {name} | {r["quant_bits"]}bit | {r["accuracy"]:.3f} |')
        lines.append('')

    # Benchmark
    if benchmark_results:
        lines.append('## 十三、系统性能基准')
        lines.append('')
        lines.append(f'| 指标 | 值 |')
        lines.append(f'|------|-----|')
        lines.append(f'| 平均延迟 | {benchmark_results["mean_ms"]:.2f} ms |')
        lines.append(f'| 中位延迟 | {benchmark_results["median_ms"]:.2f} ms |')
        lines.append(f'| P95 延迟 | {benchmark_results["p95_ms"]:.2f} ms |')
        lines.append(f'| P99 延迟 | {benchmark_results["p99_ms"]:.2f} ms |')
        lines.append(f'| 标准差 | {benchmark_results["std_ms"]:.2f} ms |')
        lines.append(f'| 等效 FPS | {benchmark_results["fps_estimate"]} |')
        lines.append(f'| 测试迭代 | {benchmark_results["iterations"]} |')
        lines.append('')

    # Recommendations
    lines.append('## 十四、改进建议')
    lines.append('')
    if pip_comparison:
        pip_f1 = pip_comparison.get('PIP敏感手势', {}).get('avg_f1', 0)
        robust_f1 = pip_comparison.get('PIP鲁棒手势', {}).get('avg_f1', 0)
        if robust_f1 > pip_f1:
            gap = (robust_f1 - pip_f1) * 100
            lines.append(f'- [WARNING] PIP-sensitive gesture F1 {gap:.1f}% lower than robust gestures. Add MCP auxiliary checks for Victory/Space/Six/Three/Four')
    if speed_results:
        fast_acc = speed_results.get('fast', {}).get('accuracy', 0)
        slow_acc = speed_results.get('slow', {}).get('accuracy', 0)
        if slow_acc > fast_acc:
            lines.append(f'- [WARNING] High-speed accuracy drops {(slow_acc - fast_acc) * 100:.1f}%. Adaptive sampling active but larger models may help.')
    if noise_results:
        severe_acc = noise_results.get('severe_jitter', {}).get('accuracy', 0)
        clean_acc = noise_results.get('clean', {}).get('accuracy', 0)
        if clean_acc > severe_acc:
            lines.append(f'- Severe noise accuracy drops {(clean_acc - severe_acc) * 100:.1f}%, consider adding keypoint smoothing filter.')
    if ece > 0.1:
        lines.append(f'- [WARNING] ECE={ece:.4f} > 0.1, confidence scoring needs recalibration.')
    lines.append('')

    report = '\n'.join(lines)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'\nReport saved: {output_path}')
    return report


def main():
    args = parse_args()

    print('=' * 60)
    print('  GESTURE RECOGNITION SYSTEM v3.0 — VALIDATION SUITE')
    print('  Methodology: Wagh et al. (2025) JNeuroEngRehabil')
    print('=' * 60)

    # Load data
    samples = load_data(args.data, args.data_dir)
    if len(samples) == 0:
        print('ERROR: No samples loaded. Check --data and --data_dir paths.')
        sys.exit(1)

    # Initialize recognizer
    from core.fallback.rule_recognizer import RuleRecognizer
    recognizer = RuleRecognizer()

    # Run recognition on all samples
    print(f'\nRunning recognition on {len(samples)} samples...')
    y_true, y_pred, confidences = [], [], []

    for i, sample in enumerate(samples):
        frame_results = recognize_sequence(sample['keypoints'], recognizer)
        gestures = [g for g, _ in frame_results]
        confs = [c for _, c in frame_results]
        pred = majority_vote(gestures)

        y_true.append(sample['label'])
        y_pred.append(pred)
        confidences.append(np.mean(confs) if confs else 0.5)

        if (i + 1) % 50 == 0:
            print(f'  Processed {i+1}/{len(samples)}...')

    # Compute metrics
    print('\n--- Per-Gesture Metrics ---')
    metrics = compute_metrics(y_true, y_pred)
    for label, m in sorted(metrics.items()):
        if label in ('macro_avg', 'accuracy'):
            continue
        pip_tag = ' [PIP敏感]' if label in PIP_SENSITIVE else ''
        print(f'  {label:16s}  P:{m["precision"]:.3f}  R:{m["recall"]:.3f}  '
              f'F1:{m["f1"]:.3f}  (n={m["support"]}){pip_tag}')

    print(f'\n  Accuracy: {metrics["accuracy"]:.3f}')
    print(f'  Macro F1: {metrics["macro_avg"]["f1"]:.3f}')

    # Confusion matrix
    print('\n--- Confusion Matrix ---')
    labels, cm, top_errors = compute_confusion_matrix(y_true, y_pred)
    confusion_data = (labels, cm, top_errors)
    if top_errors:
        print('  Top confusions:')
        for true_l, pred_l, count in top_errors[:5]:
            print(f'    {true_l} → {pred_l} ({count}x)')

    # Bootstrap CI
    print(f'\n--- Bootstrap CI (n={args.n_bootstrap}) ---')

    def accuracy_fn(data):
        t = [d[0] for d in data]
        p = [d[1] for d in data]
        return sum(1 for a, b in zip(t, p) if a == b) / len(t)

    def f1_fn(data):
        t = [d[0] for d in data]
        p = [d[1] for d in data]
        m = compute_metrics(t, p)
        return m['macro_avg']['f1']

    paired = list(zip(y_true, y_pred))
    acc_mean, acc_low, acc_high = bootstrap_ci(paired, accuracy_fn, args.n_bootstrap)
    f1_mean, f1_low, f1_high = bootstrap_ci(paired, f1_fn, args.n_bootstrap)

    ci_results = {
        'accuracy': (acc_low, acc_high),
        'f1': (f1_low, f1_high)
    }
    print(f'  Accuracy: {acc_mean:.3f} [{acc_low:.3f}, {acc_high:.3f}]')
    print(f'  Macro F1: {f1_mean:.3f} [{f1_low:.3f}, {f1_high:.3f}]')

    # PIP sensitivity analysis
    print('\n--- PIP Joint Sensitivity Analysis ---')
    pip_sensitive_samples = [(t, p) for t, p in zip(y_true, y_pred) if p in PIP_SENSITIVE]
    pip_robust_samples = [(t, p) for t, p in zip(y_true, y_pred) if p not in PIP_SENSITIVE]

    pip_comparison = {}
    if pip_sensitive_samples:
        pip_t, pip_p = zip(*pip_sensitive_samples)
        pip_m = compute_metrics(pip_t, pip_p)
        pip_f1 = pip_m['macro_avg']['f1']
        pip_comparison['PIP敏感手势'] = {'avg_f1': pip_f1, 'count': len(pip_sensitive_samples)}
        print(f'  PIP敏感手势 F1: {pip_f1:.3f} (n={len(pip_sensitive_samples)})')

    if pip_robust_samples:
        rob_t, rob_p = zip(*pip_robust_samples)
        rob_m = compute_metrics(rob_t, rob_p)
        rob_f1 = rob_m['macro_avg']['f1']
        pip_comparison['PIP鲁棒手势'] = {'avg_f1': rob_f1, 'count': len(pip_robust_samples)}
        print(f'  PIP鲁棒手势 F1: {rob_f1:.3f} (n={len(pip_robust_samples)})')

    # Speed degradation test
    speed_results = None
    if args.speed_test:
        print('\n--- Speed Degradation Test ---')
        speed_results = test_speed_degradation(recognizer, samples)
        for speed_name, result in speed_results.items():
            print(f'  {speed_name}: acc={result["accuracy"]:.3f} (n={result["n_samples"]})')

    # Noise robustness test
    noise_results = None
    if args.noise_test:
        print('\n--- Noise Robustness Test ---')
        noise_results = test_noise_robustness(recognizer, samples)
        for noise_name, result in noise_results.items():
            print(f'  {noise_name}: acc={result["accuracy"]:.3f} (n={result["n_samples"]})')

    # New robustness tests (v3.0)
    occlusion_results = None
    if args.occlusion_test:
        print('\n--- Occlusion Robustness Test ---')
        occlusion_results = test_occlusion_robustness(recognizer, samples)
        for occ_name, result in occlusion_results.items():
            print(f'  {occ_name}: acc={result["accuracy"]:.3f} ({result["occluded_landmarks"]} hidden)')

    lighting_results = None
    if args.lighting_test:
        print('\n--- Lighting Variability Test ---')
        lighting_results = test_lighting_variability(recognizer, samples)
        for light_name, result in lighting_results.items():
            print(f'  {light_name}: acc={result["accuracy"]:.3f} (σ={result["jitter_std"]:.4f})')

    distance_results = None
    if args.distance_test:
        print('\n--- Distance Variability Test ---')
        distance_results = test_distance_variability(recognizer, samples)
        for dist_name, result in distance_results.items():
            print(f'  {dist_name}: acc={result["accuracy"]:.3f} (scale={result["scale"]:.1f})')

    resolution_results = None
    if args.resolution_test:
        print('\n--- Resolution Degradation Test ---')
        resolution_results = test_resolution_degradation(recognizer, samples)
        for res_name, result in resolution_results.items():
            print(f'  {res_name}: acc={result["accuracy"]:.3f}')

    # Ablation study
    ablation_results = None
    if args.ablate:
        print('\n--- Ablation Study ---')
        ablation_results = run_ablation_study(recognizer, samples)
        base_acc = ablation_results['full_system']['accuracy']
        for config_name, data in ablation_results.items():
            delta = data['accuracy'] - base_acc
            delta_str = f' ({delta:+.3f})' if delta != 0 else ''
            print(f'  {config_name}: acc={data["accuracy"]:.3f}, '
                  f'F1={data["macro_f1"]:.3f}{delta_str}')

    # Performance benchmarking
    benchmark_results = None
    if args.benchmark:
        print(f'\n--- Performance Benchmark ({args.benchmark_iters} iterations) ---')
        benchmark_results = run_benchmark(recognizer, samples, args.benchmark_iters)
        print(f'  Mean: {benchmark_results["mean_ms"]:.2f} ms')
        print(f'  P95:  {benchmark_results["p95_ms"]:.2f} ms')
        print(f'  P99:  {benchmark_results["p99_ms"]:.2f} ms')
        print(f'  FPS:  {benchmark_results["fps_estimate"]}')

    # Confidence calibration
    print('\n--- Confidence Calibration ---')
    calibration, ece = test_confidence_calibration(y_true, y_pred, confidences)
    for cal in calibration:
        print(f'  Bin {cal["bin"]}: acc={cal["accuracy"]:.3f} '
              f'(expected {cal["expected_confidence"]:.2f}, n={cal["count"]})')
    print(f'  ECE: {ece:.4f}')

    # Generate reliability diagram
    plot_reliability_diagram(calibration, ece)

    # Generate report
    report_path = args.report or 'data/validation_report.md'
    report = generate_report(
        metrics, speed_results, noise_results, calibration, ece,
        pip_comparison, ci_results, report_path,
        confusion_data=confusion_data,
        occlusion_results=occlusion_results,
        lighting_results=lighting_results,
        distance_results=distance_results,
        resolution_results=resolution_results,
        ablation_results=ablation_results,
        benchmark_results=benchmark_results,
    )
    print(f'\nReport saved to: {report_path}')
    print(f'Accuracy: {metrics["accuracy"]:.3f} [{ci_results["accuracy"][0]:.3f}, {ci_results["accuracy"][1]:.3f}]')
    print(f'Macro F1: {metrics["macro_avg"]["f1"]:.3f} [{ci_results["f1"][0]:.3f}, {ci_results["f1"][1]:.3f}]')
    print(f'ECE: {ece:.4f}')
    if speed_results:
        print(f'Speed test: slow={speed_results["slow"]["accuracy"]:.3f} fast={speed_results["fast"]["accuracy"]:.3f}')
    if benchmark_results:
        print(f'Benchmark: {benchmark_results["fps_estimate"]} FPS (P95={benchmark_results["p95_ms"]:.2f}ms)')


if __name__ == '__main__':
    main()
