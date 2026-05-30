# coding:utf-8
"""Cross-dataset evaluation: test our models on processed external data.

Usage:
  python tools/eval_external.py \
      --data data/external_processed --model E3 \
      --checkpoint models/checkpoints/E3_spatial_geo/best_model.pth

  python tools/eval_external.py \
      --data data/external_processed --model E4 \
      --checkpoint models/checkpoints/E4_spatial_cnn_geo/best_model.pth

  python tools/eval_external.py \
      --data data/external_processed --model E6 \
      --checkpoint models/checkpoints/E6_semantic/best_model.pth
"""
import argparse, os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Processed data dir (with manifest.json)')
    p.add_argument('--model', required=True, choices=['E3', 'E4', 'E6'],
                   help='Which model to evaluate')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_models(ckpt_path, device):
    """Load feature extractors from checkpoint."""
    from core.feature.spatial_gcn import SpatialGCN
    from core.feature.hand_shape_context import HandShapeContext
    from core.feature.visual_encoder import LightweightVisualEncoder
    from core.feature.multimodal_fusion import CrossModalFusion, GatedFusion

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_mods = ckpt.get('modalities', {})
    use_spatial = ckpt_mods.get('spatial', 'spatial_model' in ckpt)
    use_cnn = ckpt_mods.get('cnn', 'visual_model' in ckpt)
    use_geo = ckpt_mods.get('geometric', 'geometric_model' in ckpt)
    use_gated = use_cnn and use_geo and ('gated_fusion' in ckpt)

    # Auto-detect angle_dim
    angle_dim = 64
    if use_spatial and 'spatial_model' in ckpt:
        w = ckpt['spatial_model'].get('final_fusion.0.weight')
        if w is not None:
            angle_dim = w.shape[1] - 256

    spatial = SpatialGCN(angle_dim=angle_dim).to(device).eval()
    spatial.load_state_dict(ckpt['spatial_model'])

    geometric = None
    if use_geo and 'geometric_model' in ckpt:
        geometric = HandShapeContext().to(device).eval()
        geometric.load_state_dict(ckpt['geometric_model'])

    cnn = None
    if use_cnn and 'visual_model' in ckpt:
        cnn = LightweightVisualEncoder(freeze_backbone=True).to(device).eval()
        cnn.load_state_dict(ckpt['visual_model'])

    gated = None
    if use_gated:
        gated = GatedFusion().to(device).eval()
        gated.load_state_dict(ckpt['gated_fusion'])

    vis_dim = 512 if use_cnn else 256
    fusion = CrossModalFusion(visual_dim=vis_dim, motion_dim=256).to(device).eval()
    fusion.load_state_dict(ckpt['fusion_model'])

    return {
        'spatial': spatial, 'geometric': geometric, 'cnn': cnn, 'gated': gated,
        'fusion': fusion, 'vis_dim': vis_dim,
        'flags': {'use_spatial': use_spatial, 'use_cnn': use_cnn, 'use_geo': use_geo,
                   'use_gated': use_gated},
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'Device: {device}')
    print(f'Model: {args.model}')

    models = load_models(args.checkpoint, device)
    flags = models['flags']
    print(f"Modalities: spatial={flags['use_spatial']}, cnn={flags['use_cnn']}, "
          f"geo={flags['use_geo']}, gated={flags['use_gated']}")

    # Load manifest
    manifest_path = os.path.join(args.data, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Group by class
    from collections import defaultdict
    class_samples = defaultdict(list)
    for item in manifest:
        class_samples[item['label_idx']].append(item)

    # Normalization for CNN - ImageNet stats
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    all_correct = 0
    all_total = 0
    per_class = {}

    for label_idx, items in sorted(class_samples.items()):
        correct = 0
        total = 0
        predictions = []

        for item in items:
            base = item['file']
            kp_path = os.path.join(args.data, f'cls_{label_idx:02d}', f'{base}_kp.npy')
            roi_path = os.path.join(args.data, f'cls_{label_idx:02d}', f'{base}_roi.npy')

            if not os.path.exists(kp_path) or not os.path.exists(roi_path):
                continue

            kp = np.load(kp_path)  # [21, 3]
            roi = np.load(roi_path)  # [96, 96, 3], uint8

            # Prepare inputs: kp [1, 21, 3], roi [1, 3, 96, 96]
            kp_t = torch.from_numpy(kp).float().unsqueeze(0).to(device)
            roi_t = torch.from_numpy(roi).float().permute(2, 0, 1) / 255.0  # [3, 96, 96]
            roi_t = (((roi_t - imagenet_mean) / imagenet_std).unsqueeze(0)).to(device)

            with torch.no_grad():
                spa = models['spatial'](kp_t)  # [1, 256]

                geo = torch.zeros(1, 256, device=device)
                if models['geometric']:
                    geo = models['geometric'](kp_t)

                cnn_feat = torch.zeros(1, 512, device=device)
                if models['cnn']:
                    cnn_feat = models['cnn'](roi_t)

                # Visual slot
                if models['gated']:
                    vis = models['gated'](cnn_feat, geo)
                elif flags['use_cnn']:
                    vis = cnn_feat
                elif flags['use_geo']:
                    vis = geo
                else:
                    vis = torch.zeros(1, models['vis_dim'], device=device)

                fused = models['fusion'](vis, spa, torch.zeros(1, 256, device=device))
                # TCN expects [B, T, 256], we have single frame
                # Use cross-entropy directly on fused features
                # For single frame, we can use the CrossModalFusion output directly
                # Actually, our models have SlowFastTCN after fusion. For single image
                # we can't really use TCN. Let me check...

                # The fused features are 256-dim. We need to run through the full
                # model pipeline. But TCN needs temporal context.
                # For single-frame evaluation we'll use the fused features directly
                # with a simple linear classifier.

                # Actually, let's use the full pipeline: replicate single frame
                # T times to form a pseudo-sequence
                T = 16  # typical sequence length
                fused_seq = fused.unsqueeze(0).expand(-1, T, -1)  # [1, 16, 256]

                # Run TCN
                from core.temporal.slowfast_tcn import SlowFastTCN
                if not hasattr(main, '_tcn'):
                    main._tcn = SlowFastTCN(input_dim=256, num_classes=10).to(device).eval()
                    # Load TCN weights from same checkpoint
                    full_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
                    if 'tcn_model' in full_ckpt:
                        main._tcn.load_state_dict(full_ckpt['tcn_model'])

                logits = main._tcn(fused_seq).mean(dim=1)  # [1, 10]
                pred = logits.argmax(dim=1).item()

            total += 1
            if pred == label_idx:
                correct += 1
            predictions.append(pred)

        per_class[label_idx] = {'correct': correct, 'total': total, 'acc': correct / total}
        print(f'  Class {label_idx}: {correct}/{total} ({correct/total*100:.1f}%) - preds: {predictions[:5]}...')

    all_correct = sum(v['correct'] for v in per_class.values())
    all_total = sum(v['total'] for v in per_class.values())

    print(f'\n{"="*50}')
    print(f'Overall: {all_correct}/{all_total} ({all_correct/all_total*100:.1f}%)')
    print(f'Per-class:')
    for idx, info in sorted(per_class.items()):
        bar = '█' * int(info['acc'] * 30) + '░' * (30 - int(info['acc'] * 30))
        print(f'  Class {idx:2d}: {info["acc"]*100:5.1f}% {bar} ({info["correct"]}/{info["total"]})')

    # Save results
    result_path = os.path.join(args.data, f'cross_dataset_{args.model}_result.json')
    with open(result_path, 'w') as f:
        json.dump({
            'model': args.model, 'accuracy': all_correct / all_total,
            'per_class': {str(k): v for k, v in per_class.items()},
            'total_samples': all_total,
        }, f, indent=2)
    print(f'Saved: {result_path}')


if __name__ == '__main__':
    main._tcn = None
    main()
