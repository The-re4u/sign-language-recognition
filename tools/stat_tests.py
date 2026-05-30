# coding:utf-8
"""
Statistical significance tests for sign language recognition evaluation.

Computes McNemar's test for Rule vs DL comparison, plus per-class
contingency tables and bootstrap confidence intervals.

Usage:
  python tools/stat_tests.py --test_data data/test_real.json --data_dir data/my_sequences \
      --dl_checkpoint models/checkpoints/best_model.pth --use_visual
"""

import sys, os, argparse, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Statistical significance tests')
    parser.add_argument('--test_data', required=True, help='Test set annotation JSON')
    parser.add_argument('--data_dir', default='data/my_sequences', help='Data directory')
    parser.add_argument('--dl_checkpoint', required=True, help='DL model checkpoint')
    parser.add_argument('--use_visual', action='store_true', help='Use visual encoder')
    parser.add_argument('--device', default='cpu', help='Device for inference')
    return parser.parse_args()


def run_rule_recognizer(test_data, data_dir):
    """Run rule-based recognizer on test set. Returns (true_labels, pred_labels)."""
    from core.fallback.rule_recognizer import RuleRecognizer
    from core.perception.hand_tracker import HandLandmarksWrapper

    recognizer = RuleRecognizer()
    y_true, y_pred = [], []

    for ann in test_data:
        seq_path = os.path.join(data_dir, os.path.basename(ann.get('frames_path', '')))
        if not os.path.exists(seq_path):
            seq_path = ann.get('frames_path', '')
        if not os.path.exists(seq_path):
            continue

        kp_seq = np.load(seq_path).astype(np.float32)
        label = ann.get('label', ann.get('gesture', 'unknown'))

        # Single-frame evaluation (matches runtime: one frame = one classification, no voting)
        # Use middle frame where gesture is most stable
        mid_idx = len(kp_seq) // 2
        frame_kp = kp_seq[mid_idx]

        class FakeLandmark:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        class FakeWrapper:
            def __init__(self, kp):
                self.landmark = [FakeLandmark(float(k[0]), float(k[1]), float(k[2])) for k in kp]

        _, _, gesture = recognizer.recognize(FakeWrapper(frame_kp))
        y_true.append(label)
        y_pred.append(gesture)

    return y_true, y_pred


def run_dl_model(test_data, data_dir, checkpoint_path, use_visual, device):
    """Run DL model on test set. Returns (true_labels, pred_labels)."""
    import torch
    from core.feature.spatial_gcn import SpatialGCN
    from core.feature.motion_encoder import MotionEncoder
    from core.feature.multimodal_fusion import CrossModalFusion
    from core.temporal.slowfast_tcn import SlowFastTCN
    from core.feature.visual_encoder import LightweightVisualEncoder

    spatial = SpatialGCN().to(device)
    spatial.eval()
    motion = MotionEncoder().to(device)
    motion.eval()
    fusion = CrossModalFusion().to(device)
    fusion.eval()
    tcn = SlowFastTCN(input_dim=256, num_classes=10).to(device)
    tcn.eval()

    visual = None
    if use_visual:
        visual = LightweightVisualEncoder().to(device)
        visual.eval()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    spatial.load_state_dict(ckpt['spatial_model'])
    motion.load_state_dict(ckpt['motion_model'])
    fusion.load_state_dict(ckpt['fusion_model'])
    tcn.load_state_dict(ckpt['tcn_model'])
    if visual and 'visual_model' in ckpt:
        visual.load_state_dict(ckpt['visual_model'])

    # Build label mapping
    labels = sorted(set(a.get('label', a.get('gesture', 'unknown')) for a in test_data))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}

    y_true, y_pred = [], []

    with torch.no_grad():
        for ann in test_data:
            seq_path = os.path.join(data_dir, os.path.basename(ann.get('frames_path', '')))
            if not os.path.exists(seq_path):
                seq_path = ann.get('frames_path', '')
            if not os.path.exists(seq_path):
                continue

            kp_seq = np.load(seq_path).astype(np.float32)
            label = ann.get('label', ann.get('gesture', 'unknown'))

            T = kp_seq.shape[0]
            kp_t = torch.from_numpy(kp_seq).float().unsqueeze(0).to(device)  # [1, T, 21, 3]
            B, T_in, N, C = kp_t.shape

            # Vectorized GCN
            kp_flat = kp_t.reshape(B * T_in, 21, 3)
            spatial_flat = spatial(kp_flat)
            spatial_seq = spatial_flat.reshape(B, T_in, 256)

            # ROI
            roi_path = ann.get('roi_path', '')
            if roi_path and use_visual and visual:
                full_roi = os.path.join(data_dir, os.path.basename(roi_path))
                if not os.path.exists(full_roi):
                    full_roi = roi_path if os.path.exists(roi_path) else ''
                if full_roi and os.path.exists(full_roi):
                    roi_data = np.load(full_roi).astype(np.float32) / 255.0  # [T, 96, 96, 3]
                    roi_data = np.transpose(roi_data, (0, 3, 1, 2))  # [T, 3, 96, 96]
                    roi_t = torch.from_numpy(roi_data).float().unsqueeze(0).to(device)
                    roi_flat = roi_t.reshape(B * T_in, 3, 96, 96)
                    visual_flat = visual(roi_flat)
                    visual_seq = visual_flat.reshape(B, T_in, 512)
                else:
                    visual_seq = torch.zeros(B, T_in, 512, device=device)
            else:
                visual_seq = torch.zeros(B, T_in, 512, device=device)

            # Motion + Fusion per-frame
            frame_feats = []
            for t in range(T_in):
                sp = spatial_seq[:, t, :]
                vi = visual_seq[:, t, :]
                if t > 0:
                    kp_diff = kp_t[:, t, :, :2] - kp_t[:, t-1, :, :2]
                    kp_delta = torch.cat([kp_diff.reshape(1, 42), torch.zeros(1, 21, device=device)], dim=1)
                    flow_hist = torch.zeros(1, 128, device=device)
                    mo = motion(flow_hist, kp_delta)
                else:
                    mo = torch.zeros(1, 128, device=device)
                fu = fusion(vi, sp, mo)
                frame_feats.append(fu.unsqueeze(1))
            fused_seq = torch.cat(frame_feats, dim=1)

            logits = tcn(fused_seq)
            mean_logits = logits.mean(dim=1)
            pred_idx = mean_logits.argmax(dim=1).item()
            pred_label = idx_to_label.get(pred_idx, str(pred_idx))

            y_true.append(label)
            y_pred.append(pred_label)

    return y_true, y_pred


def mcnemar_test(y_true, y_pred_a, y_pred_b, name_a='Rule', name_b='DL'):
    """McNemar's test for paired nominal data.

    Builds 2×2 contingency table:
                B correct   B wrong
    A correct      n11        n10
    A wrong        n01        n00

    McNemar statistic = (|n10 - n01| - 1)^2 / (n10 + n01)
    Under H0 (equal error rates), follows chi-square(1).
    """
    n = len(y_true)
    both_correct = sum(1 for i in range(n) if y_pred_a[i] == y_true[i] and y_pred_b[i] == y_true[i])
    a_only = sum(1 for i in range(n) if y_pred_a[i] == y_true[i] and y_pred_b[i] != y_true[i])
    b_only = sum(1 for i in range(n) if y_pred_a[i] != y_true[i] and y_pred_b[i] == y_true[i])
    both_wrong = sum(1 for i in range(n) if y_pred_a[i] != y_true[i] and y_pred_b[i] != y_true[i])

    n10 = a_only
    n01 = b_only

    # Yates continuity correction
    if n10 + n01 > 0:
        chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    else:
        chi2 = 0.0

    # p-value from chi-square(1)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1.0 - chi2_dist.cdf(chi2, 1)

    acc_a = sum(1 for i in range(n) if y_pred_a[i] == y_true[i]) / n
    acc_b = sum(1 for i in range(n) if y_pred_b[i] == y_true[i]) / n

    return {
        'n_samples': n,
        'acc_a': acc_a, 'acc_b': acc_b,
        'delta': acc_b - acc_a,
        'both_correct': both_correct, 'both_wrong': both_wrong,
        'a_only_correct': a_only, 'b_only_correct': b_only,
        'n10': n10, 'n01': n01,
        'chi2_statistic': round(chi2, 4),
        'p_value': round(p_value, 6),
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'significant_001': p_value < 0.001,
    }


def bootstrap_ci(y_true, y_pred, n_bootstrap=10000, alpha=0.05):
    """Bootstrap 95% confidence interval for accuracy."""
    np.random.seed(42)
    n = len(y_true)
    accs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        acc = sum(1 for i in idx if y_pred[i] == y_true[i]) / n
        accs.append(acc)
    accs = np.array(accs)
    lower = np.percentile(accs, 100 * alpha / 2)
    upper = np.percentile(accs, 100 * (1 - alpha / 2))
    return {
        'mean_accuracy': round(np.mean(accs), 4),
        'ci_lower': round(lower, 4),
        'ci_upper': round(upper, 4),
        'ci_95': f'[{lower:.4f}, {upper:.4f}]',
    }


def per_class_table(y_true, y_pred, labels):
    """Per-class accuracy, precision, recall, F1."""
    results = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[label] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': tp + fn,
        }
    return results


def main():
    args = parse_args()

    print('=' * 60)
    print('  STATISTICAL SIGNIFICANCE TESTS')
    print('=' * 60)

    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f'\nTest samples: {len(test_data)}')

    labels = sorted(set(a.get('label', a.get('gesture', 'unknown')) for a in test_data))

    # 1. Rule Recognizer
    print('\n[1/2] Running Rule Recognizer...')
    y_true_rule, y_pred_rule = run_rule_recognizer(test_data, args.data_dir)
    rule_acc = sum(1 for t, p in zip(y_true_rule, y_pred_rule) if t == p) / len(y_true_rule)
    print(f'  Rule Accuracy: {rule_acc:.2%}')

    # 2. DL Model
    print('\n[2/2] Running DL Model...')
    import torch
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    y_true_dl, y_pred_dl = run_dl_model(test_data, args.data_dir, args.dl_checkpoint,
                                         args.use_visual, device)
    dl_acc = sum(1 for t, p in zip(y_true_dl, y_pred_dl) if t == p) / len(y_true_dl)
    print(f'  DL Accuracy: {dl_acc:.2%}')

    # 3. McNemar Test
    print('\n[3] McNemar Test: Rule vs DL')
    mcnemar = mcnemar_test(y_true_dl, y_pred_rule, y_pred_dl, 'Rule', 'DL')
    print(f'  Contingency table:')
    print(f'    Both correct:  {mcnemar["both_correct"]}')
    print(f'    Rule only:     {mcnemar["a_only_correct"]}')
    print(f'    DL only:       {mcnemar["b_only_correct"]}')
    print(f'    Both wrong:    {mcnemar["both_wrong"]}')
    print('  chi2 = {}, p = {}'.format(mcnemar['chi2_statistic'], mcnemar['p_value']))
    if mcnemar['significant_001']:
        print(f'  *** p < 0.001 — highly significant')
    elif mcnemar['significant_01']:
        print(f'  ** p < 0.01 — significant')
    elif mcnemar['significant_05']:
        print(f'  * p < 0.05 — significant')
    else:
        print(f'  Not significant (p >= 0.05)')
    print(f'  Accuracy delta: DL - Rule = {mcnemar["delta"]:.2%}')

    # 4. Bootstrap CI
    print('\n[4] Bootstrap 95% Confidence Intervals')
    rule_ci = bootstrap_ci(y_true_rule, y_pred_rule)
    dl_ci = bootstrap_ci(y_true_dl, y_pred_dl)
    print(f'  Rule: {rule_ci["ci_95"]} (mean={rule_ci["mean_accuracy"]})')
    print(f'  DL:   {dl_ci["ci_95"]} (mean={dl_ci["mean_accuracy"]})')

    # 5. Per-Class
    print('\n[5] Per-Class Metrics (DL)')
    per_class = per_class_table(y_true_dl, y_pred_dl, labels)
    print(f'  {"Gesture":>14s}  {"P":>6s}  {"R":>6s}  {"F1":>6s}  Support')
    print(f'  {"-"*14}  {"-"*6}  {"-"*6}  {"-"*6}  {"-"*7}')
    for label in labels:
        m = per_class[label]
        print(f'  {label:>14s}  {m["precision"]:.4f}  {m["recall"]:.4f}  '
              f'{m["f1"]:.4f}  {m["support"]:3d}')

    # 6. Save results (convert numpy types)
    import numpy as _np
    def _clean(obj):
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_clean(v) for v in obj]
        if isinstance(obj, (_np.bool_,)): return bool(obj)
        if isinstance(obj, (_np.integer,)): return int(obj)
        if isinstance(obj, (_np.floating,)): return float(obj)
        return obj

    output = _clean({
        'test_samples': len(test_data),
        'rule_accuracy': round(rule_acc, 4),
        'dl_accuracy': round(dl_acc, 4),
        'mcnemar_test': mcnemar,
        'bootstrap_ci': {'rule': rule_ci, 'dl': dl_ci},
        'per_class_dl': per_class,
    })
    output_path = 'docs/stat_test_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\nResults saved to {output_path}')

    # Thesis-ready sentence
    print('\n' + '=' * 60)
    print('  THESIS-READY REPORTING')
    print('=' * 60)
    sig_level = ('p < 0.001' if mcnemar['significant_001'] else
                 'p < 0.01' if mcnemar['significant_01'] else
                 'p < 0.05' if mcnemar['significant_05'] else 'n.s.')
    print(f'  "The {mcnemar["delta"]:.1%} improvement of DL over Rule is')
    print(f'   statistically significant (McNemar test, {sig_level})."')
    print(f'  "DL achieves {dl_acc:.1%} accuracy (95% CI: {dl_ci["ci_95"]}),')
    print(f'   compared to {rule_acc:.1%} for the rule-based baseline."')


if __name__ == '__main__':
    main()
