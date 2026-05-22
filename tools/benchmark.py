# coding:utf-8
"""
System performance profiling tool v3.0.

Measures end-to-end and per-module latency for the sign language recognition pipeline.
Exports profiling report in markdown format.

Usage:
  python tools/benchmark.py                          # Quick benchmark with synthetic data
  python tools/benchmark.py --iterations 500         # Longer benchmark
  python tools/benchmark.py --report benchmark.md    # Export report
  python tools/benchmark.py --module_breakdown       # Per-module timing breakdown
"""
import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description='System performance benchmarking')
    parser.add_argument('--iterations', type=int, default=200, help='Benchmark iterations')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup iterations')
    parser.add_argument('--report', type=str, default=None, help='Output markdown report path')
    parser.add_argument('--module_breakdown', action='store_true',
                        help='Enable per-module timing breakdown')
    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU delegate')
    parser.add_argument('--sequence_length', type=int, default=32,
                        help='Frames per test sequence')
    return parser.parse_args()


def benchmark_rule_recognizer(iters=200, warmup=20):
    """Benchmark RuleRecognizer on synthetic keypoints."""
    from core.fallback.rule_recognizer import RuleRecognizer
    from core.perception.hand_tracker import HandLandmarksWrapper
    from tools.generate_synthetic_data import create_canonical_hand

    class FakeLandmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class FakeWrapper:
        def __init__(self, kp):
            self.landmark = [FakeLandmark(float(k[0]), float(k[1]), float(k[2])) for k in kp]

    rec_std = RuleRecognizer()
    rec_conf = RuleRecognizer()

    # Generate diverse keypoint patterns
    patterns = [
        [False]*5,                          # Closed_Fist
        [True, False, False, False, False],  # Good
        [True, False, False, False, True],   # Space
        [True, True, True, True, True],      # Open_Palm
        [False, True, True, False, False],   # Victory
        [False, False, True, True, True],    # Three
    ]

    kp_list = [create_canonical_hand(p, 0.0) for p in patterns]

    # Warmup
    for _ in range(warmup):
        kp = kp_list[_ % len(kp_list)]
        wrapper = FakeWrapper(kp)
        rec_std.recognize(wrapper)
        rec_conf.recognize_with_confidence(wrapper)

    # Benchmark: standard recognize
    times_std = []
    for i in range(iters):
        kp = kp_list[i % len(kp_list)]
        wrapper = FakeWrapper(kp)
        t0 = time.perf_counter()
        rec_std.recognize(wrapper)
        times_std.append((time.perf_counter() - t0) * 1000)

    # Benchmark: recognize_with_confidence
    times_conf = []
    for i in range(iters):
        kp = kp_list[i % len(kp_list)]
        wrapper = FakeWrapper(kp)
        t0 = time.perf_counter()
        rec_conf.recognize_with_confidence(wrapper)
        times_conf.append((time.perf_counter() - t0) * 1000)

    return {
        'recognize': _stats(times_std, iters),
        'recognize_with_confidence': _stats(times_conf, iters),
    }


def benchmark_hand_tracker(iters=50, warmup=10, use_gpu=True):
    """Benchmark MediaPipe HandTracker with a dummy image."""
    try:
        from core.perception.hand_tracker import HandTracker
    except Exception as e:
        return {'error': str(e)}

    import cv2
    tracker = HandTracker(use_gpu=use_gpu)
    dummy = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    dummy_rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)

    for _ in range(warmup):
        tracker.detect_image(dummy_rgb)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        tracker.detect_image(dummy_rgb)
        times.append((time.perf_counter() - t0) * 1000)

    tracker.close()
    return _stats(times, iters)


def benchmark_kinematic_features(iters=500, warmup=50):
    """Benchmark kinematic feature computation."""
    from core.feature.kinematic_features import compute_all_kinematic_features
    from tools.generate_synthetic_data import create_canonical_hand

    prev_kp = create_canonical_hand([False]*5, 0.0)
    curr_kp = create_canonical_hand([True, False, False, False, True], 0.0)

    for _ in range(warmup):
        compute_all_kinematic_features(prev_kp, curr_kp, fps=30, kp_history=[prev_kp[:, :2]]*16)

    times = []
    history = [prev_kp[:, :2]] * 16
    for _ in range(iters):
        t0 = time.perf_counter()
        compute_all_kinematic_features(prev_kp, curr_kp, fps=30, kp_history=history)
        times.append((time.perf_counter() - t0) * 1000)

    return _stats(times, iters)


def benchmark_semantic_parser(iters=200, warmup=20):
    """Benchmark TemporalSemanticParser."""
    from core.semantic.temporal_parser import TemporalSemanticParser

    parser = TemporalSemanticParser('config/semantic_chains.json')
    tokens = ['One', 'Good', 'Eight', 'Open_Palm', 'Closed_Fist']

    for _ in range(warmup):
        parser.flush()
        parser.add_token(tokens[_ % 5])

    times = []
    for i in range(iters):
        t0 = time.perf_counter()
        parser.add_token(tokens[i % 5])
        times.append((time.perf_counter() - t0) * 1000)

    return _stats(times, iters)


def benchmark_end_to_end_sequence(iters=50, seq_len=32):
    """Simulate end-to-end processing of a full gesture sequence."""
    from core.fallback.rule_recognizer import RuleRecognizer
    from core.semantic.temporal_parser import TemporalSemanticParser
    from tools.generate_synthetic_data import create_canonical_hand

    rec = RuleRecognizer()
    parser = TemporalSemanticParser('config/semantic_chains.json')

    class FakeLandmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class FakeWrapper:
        def __init__(self, kp):
            self.landmark = [FakeLandmark(float(k[0]), float(k[1]), float(k[2])) for k in kp]

    patterns = [
        ([False]*5, 'Closed_Fist'),
        ([True, False, False, False, False], 'Good'),
        ([True, True, True, True, True], 'Open_Palm'),
    ]

    total_times = []
    for _ in range(iters):
        parser.flush()
        seq = []
        for t in range(seq_len):
            pattern_idx = (t // 10) % len(patterns)
            kp = create_canonical_hand(patterns[pattern_idx][0], 0.0)
            seq.append(kp)

        t0 = time.perf_counter()
        for kp in seq:
            wrapper = FakeWrapper(kp)
            gesture = rec.recognize(wrapper)[2]
            parser.add_token(gesture)
        total_times.append((time.perf_counter() - t0) * 1000)

    return _stats(total_times, iters)


def _stats(times_ms, iters):
    t = np.array(times_ms)
    return {
        'mean_ms': round(float(np.mean(t)), 3),
        'median_ms': round(float(np.median(t)), 3),
        'p95_ms': round(float(np.percentile(t, 95)), 3),
        'p99_ms': round(float(np.percentile(t, 99)), 3),
        'std_ms': round(float(np.std(t)), 3),
        'min_ms': round(float(np.min(t)), 3),
        'max_ms': round(float(np.max(t)), 3),
        'iterations': iters,
        'fps_equivalent': round(1000.0 / np.mean(t), 1),
    }


def generate_report(all_results, output_path=None):
    lines = ['# 手势识别系统 v3.0 — 性能基准报告', '',
             f'*生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}*', '',
             '## 一、总览', '',
             '| 模块 | 平均延迟 | P95 延迟 | 等效 FPS |',
             '|------|---------|---------|----------|']

    for module_name, result in all_results.items():
        if isinstance(result, dict) and 'mean_ms' in result:
            lines.append(f'| {module_name} | {result["mean_ms"]:.2f} ms | '
                         f'{result["p95_ms"]:.2f} ms | {result["fps_equivalent"]} |')

    lines.append('')
    lines.append('## 二、各模块详细指标')
    lines.append('')

    for module_name, result in all_results.items():
        if isinstance(result, dict):
            if 'error' in result:
                lines.append(f'### {module_name}')
                lines.append(f'**错误**: {result["error"]}')
                lines.append('')
                continue
            if 'recognize' in result:
                for sub_name, sub_result in result.items():
                    lines.append(f'### {module_name}.{sub_name}')
                    lines.extend(_stats_table(sub_result))
            else:
                lines.append(f'### {module_name}')
                lines.extend(_stats_table(result))

    lines.append('## 三、系统瓶颈分析')
    lines.append('')
    modules_with_latency = {}
    for name, r in all_results.items():
        if isinstance(r, dict):
            if 'mean_ms' in r:
                modules_with_latency[name] = r['mean_ms']
            elif 'recognize' in r:
                for sn, sr in r.items():
                    if 'mean_ms' in sr:
                        modules_with_latency[f'{name}/{sn}'] = sr['mean_ms']

    if modules_with_latency:
        total_latency = sum(modules_with_latency.values())
        lines.append(f'**理论总延迟**: {total_latency:.2f} ms')
        lines.append('')
        lines.append('| 模块 | 延迟占比 |')
        lines.append('|------|---------|')
        sorted_modules = sorted(modules_with_latency.items(), key=lambda x: -x[1])
        for name, latency in sorted_modules:
            pct = latency / total_latency * 100
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            lines.append(f'| {name} | {bar} {pct:.1f}% ({latency:.2f}ms) |')

    lines.append('')
    report = '\n'.join(lines)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'Report saved: {output_path}')
    return report


def _stats_table(result):
    lines = []
    lines.append('')
    lines.append('| 指标 | 值 |')
    lines.append('|------|-----|')
    for k in ['mean_ms', 'median_ms', 'p95_ms', 'p99_ms', 'std_ms', 'min_ms', 'max_ms', 'fps_equivalent']:
        if k in result:
            label = k.replace('_ms', ' (ms)').replace('_', ' ')
            lines.append(f'| {label} | {result[k]} |')
    lines.append('')
    return lines


def main():
    args = parse_args()

    print('=' * 60)
    print('  SYSTEM PERFORMANCE BENCHMARK v3.0')
    print(f'  Iterations: {args.iterations}, Warmup: {args.warmup}')
    print('=' * 60)

    all_results = {}

    # 1. Rule recognizer
    print('\n[1/5] Benchmarking RuleRecognizer...')
    all_results['RuleRecognizer'] = benchmark_rule_recognizer(args.iterations, args.warmup)
    r = all_results['RuleRecognizer']['recognize']
    print(f'  recognize: {r["mean_ms"]:.3f}ms mean, {r["fps_equivalent"]} FPS')
    r = all_results['RuleRecognizer']['recognize_with_confidence']
    print(f'  recognize_with_confidence: {r["mean_ms"]:.3f}ms mean, {r["fps_equivalent"]} FPS')

    # 2. Hand tracker
    print('\n[2/5] Benchmarking HandTracker (MediaPipe)...')
    all_results['HandTracker'] = benchmark_hand_tracker(
        min(50, args.iterations // 4), args.warmup // 4, args.use_gpu)
    if 'error' not in all_results['HandTracker']:
        ht = all_results['HandTracker']
        print(f'  detect_image: {ht["mean_ms"]:.2f}ms mean, {ht["fps_equivalent"]} FPS')

    # 3. Kinematic features
    print('\n[3/5] Benchmarking KinematicFeatures...')
    all_results['KinematicFeatures'] = benchmark_kinematic_features(
        args.iterations * 2, args.warmup * 2)
    kf = all_results['KinematicFeatures']
    print(f'  compute_all: {kf["mean_ms"]:.4f}ms mean')

    # 4. Semantic parser
    print('\n[4/5] Benchmarking SemanticParser...')
    all_results['SemanticParser'] = benchmark_semantic_parser(args.iterations, args.warmup)
    sp = all_results['SemanticParser']
    print(f'  add_token: {sp["mean_ms"]:.4f}ms mean')

    # 5. End-to-end sequence
    print('\n[5/5] Benchmarking End-to-End sequence...')
    all_results['EndToEndSequence'] = benchmark_end_to_end_sequence(
        min(args.iterations // 4, 50), args.sequence_length)
    e2e = all_results['EndToEndSequence']
    print(f'  {args.sequence_length}-frame sequence: {e2e["mean_ms"]:.2f}ms mean')
    print(f'  Per-frame: {e2e["mean_ms"]/args.sequence_length:.2f}ms')

    # Summary
    print('\n' + '=' * 60)
    print('  SUMMARY')
    print('=' * 60)
    total_per_frame = (
        all_results.get('HandTracker', {}).get('mean_ms', 0) +
        all_results.get('RuleRecognizer', {}).get('recognize', {}).get('mean_ms', 0) +
        all_results.get('KinematicFeatures', {}).get('mean_ms', 0) +
        all_results.get('SemanticParser', {}).get('mean_ms', 0)
    )
    print(f'  Theoretical per-frame latency: {total_per_frame:.2f} ms')
    print(f'  Theoretical max FPS: {1000/total_per_frame:.1f}' if total_per_frame > 0 else '  N/A')
    print(f'  End-to-end {args.sequence_length}-frame sequence: '
          f'{all_results["EndToEndSequence"]["mean_ms"]:.2f} ms')

    # Report
    if args.report:
        generate_report(all_results, args.report)


if __name__ == '__main__':
    main()
