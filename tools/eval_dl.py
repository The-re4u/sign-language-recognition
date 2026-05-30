# coding:utf-8
"""DL model evaluation on test sets — supports v6.0 multi-modal configurations.

Auto-detects modality configuration from checkpoint. CLI flags can disable
specific modalities for ablation evaluation.
"""
import sys, os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.feature.spatial_gcn import SpatialGCN
from core.feature.multimodal_fusion import CrossModalFusion
from core.feature.visual_encoder import LightweightVisualEncoder
from core.temporal.slowfast_tcn import SlowFastTCN
from utils.data_utils import SignLanguageDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/test.json')
    p.add_argument('--data_dir', default='data/my_sequences')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch_size', type=int, default=4)
    # Override flags (disable modalities for ablation)
    p.add_argument('--no_cnn', action='store_true', help='Disable CNN visual encoder')
    p.add_argument('--no_spatial', action='store_true', help='Disable SpatialGCN')
    p.add_argument('--no_geometric', action='store_true', help='Disable HandShapeContext')
    p.add_argument('--no_temporal', action='store_true', help='Disable TemporalEncoder')
    p.add_argument('--no_motion', action='store_true', help='Disable legacy motion encoder')
    # Legacy flags
    p.add_argument('--use_visual', action='store_true', help='(legacy) Enable CNN on old checkpoints')
    p.add_argument('--no_visual', action='store_true', help='(legacy) Disable CNN')
    p.add_argument('--no_gcn', action='store_true', help='(legacy) Disable spatial GCN')
    # Output
    p.add_argument('--output_confusion', action='store_true', help='Save confusion matrix CSV')
    return p.parse_args()


def infer_fusion_dims(fusion_state):
    """Infer visual_dim and motion_dim from saved fusion projection weights."""
    vis_dim = fusion_state['vis_proj.0.weight'].shape[1]   # [256, vis_dim]
    mot_dim = fusion_state['mot_proj.0.weight'].shape[1]   # [256, mot_dim]
    return vis_dim, mot_dim


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_mods = ckpt.get('modalities', {})
    is_new_ckpt = bool(ckpt_mods)

    # ── Determine active modalities ──
    if is_new_ckpt:
        use_spatial = ckpt_mods.get('spatial', True) and not args.no_spatial and not args.no_gcn
        use_cnn = ckpt_mods.get('cnn', True) and not args.no_cnn and not args.no_visual
        use_geometric = ckpt_mods.get('geometric', False) and not args.no_geometric
        use_temporal = ckpt_mods.get('temporal', False) and not args.no_temporal
        use_motion = False  # new scheme: temporal replaces motion
    else:
        # Old checkpoint: infer from keys present
        use_spatial = 'spatial_model' in ckpt and not args.no_gcn
        use_cnn = ('visual_model' in ckpt or args.use_visual) and not args.no_visual and not args.no_cnn
        use_geometric = False
        use_temporal = False
        use_motion = ('motion_model' in ckpt) and not args.no_motion

    vis_dim, mot_dim = infer_fusion_dims(ckpt['fusion_model'])
    num_classes = ckpt['tcn_model']['output_proj.3.weight'].shape[0]

    use_semantic = ckpt_mods.get('semantic', 'semantic_projector' in ckpt)

    print(f'Checkpoint: {"v6.0" if is_new_ckpt else "legacy"}')
    print(f'Modalities: spatial={use_spatial} cnn={use_cnn} geometric={use_geometric} '
          f'temporal={use_temporal} semantic={use_semantic} motion(legacy)={use_motion}')
    print(f'Fusion dims: vis={vis_dim} mot={mot_dim}  Classes: {num_classes}')

    # ── Auto-detect angle_dim from checkpoint (E1 uses 128) ──
    angle_dim = 64
    if use_spatial and 'spatial_model' in ckpt:
        w = ckpt['spatial_model'].get('final_fusion.0.weight')
        if w is not None:
            angle_dim = w.shape[1] - 256  # final_fusion input = 256(spatial) + angle_dim

    # ── Build models ──
    spatial_model = SpatialGCN(angle_dim=angle_dim).to(device).eval() if use_spatial else None
    cnn_model = LightweightVisualEncoder().to(device).eval() if use_cnn else None
    geometric_model = None
    temporal_model = None
    motion_model = None

    if use_geometric:
        from core.feature.hand_shape_context import HandShapeContext
        geometric_model = HandShapeContext().to(device).eval()

    if use_temporal:
        from core.feature.temporal_encoder import TemporalEncoder
        temporal_model = TemporalEncoder().to(device).eval()

    if use_motion:
        from core.feature.motion_encoder import MotionEncoder
        motion_model = MotionEncoder().to(device).eval()

    fusion_model = CrossModalFusion(visual_dim=vis_dim, motion_dim=mot_dim).to(device).eval()
    tcn_model = SlowFastTCN(input_dim=256, num_classes=num_classes).to(device).eval()

    # Geometric → visual slot gated fusion (when both CNN + geometric active)
    gated_fusion = None
    if use_cnn and use_geometric:
        from core.feature.multimodal_fusion import GatedFusion
        gated_fusion = GatedFusion().to(device).eval()

    # Semantic projector
    semantic_projector = None
    if use_semantic:
        from core.feature.semantic_encoder import SemanticProjector
        semantic_projector = SemanticProjector().to(device).eval()

    # ── Load weights ──
    if spatial_model and 'spatial_model' in ckpt:
        spatial_model.load_state_dict(ckpt['spatial_model'])
    if cnn_model and 'visual_model' in ckpt:
        cnn_model.load_state_dict(ckpt['visual_model'])
    if geometric_model and 'geometric_model' in ckpt:
        geometric_model.load_state_dict(ckpt['geometric_model'])
    if temporal_model and 'temporal_model' in ckpt:
        temporal_model.load_state_dict(ckpt['temporal_model'])
    if motion_model and 'motion_model' in ckpt:
        motion_model.load_state_dict(ckpt['motion_model'])
    # Load gated_fusion: new key first, then fall back to legacy geo_vis_proj key
    if gated_fusion and 'gated_fusion' in ckpt:
        gated_fusion.load_state_dict(ckpt['gated_fusion'])
    elif gated_fusion and 'geo_vis_proj' in ckpt:
        print('  (legacy geo_vis_proj in checkpoint — loading as best-effort, re-train for gated fusion)')
    if semantic_projector and 'semantic_projector' in ckpt:
        semantic_projector.load_state_dict(ckpt['semantic_projector'])
    fusion_model.load_state_dict(ckpt['fusion_model'])
    tcn_model.load_state_dict(ckpt['tcn_model'])
    print(f'Loaded: {args.checkpoint}')

    # ── Data ──
    dataset = SignLanguageDataset(args.data, data_dir=args.data_dir, use_cnn=use_cnn)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    print(f'Classes: {dataset.num_classes}, samples: {len(dataset)}')

    # ── Evaluate ──
    all_preds, all_labels = [], []
    with torch.no_grad():
        for kp, roi, flow, lbl, ln in loader:
            kp, roi, lbl = kp.to(device), roi.to(device), lbl.to(device)
            B, T = kp.shape[:2]

            # Spatial GCN
            if use_spatial and spatial_model:
                kp_flat = kp.reshape(B * T, 21, 3)
                spatial_seq = spatial_model(kp_flat).reshape(B, T, 256)
            else:
                spatial_seq = torch.zeros(B, T, 256, device=device)

            # Geometric (HandShapeContext)
            if use_geometric and geometric_model:
                geometric_seq = geometric_model(kp)  # [B, T, 256]
            else:
                geometric_seq = torch.zeros(B, T, 256, device=device)

            # Temporal (trajectory encoder)
            if use_temporal and temporal_model:
                temporal_seq = temporal_model(kp)  # [B, T, 256]
            else:
                temporal_seq = torch.zeros(B, T, 256, device=device)

            # CNN visual
            if use_cnn and cnn_model:
                roi_flat = roi.reshape(B * T, 3, 96, 96)
                cnn_seq = cnn_model(roi_flat).reshape(B, T, 512)
            else:
                cnn_seq = torch.zeros(B, T, 512, device=device)

            # ── Slot assignment (matches train.py) ──
            # Slot 0 (visual): CNN(512) + gated fusion if both active
            if use_cnn and use_geometric and gated_fusion is not None:
                cnn_flat = cnn_seq.reshape(B * T, 512)
                geo_flat = geometric_seq.reshape(B * T, 256)
                vis_slot = gated_fusion(cnn_flat, geo_flat).reshape(B, T, 512)
            elif use_cnn:
                vis_slot = cnn_seq  # [B, T, 512]
            elif use_geometric:
                vis_slot = geometric_seq  # [B, T, 256]
            else:
                vis_slot = torch.zeros(B, T, vis_dim, device=device)

            # Slot 1 (spatial)
            spa_slot = spatial_seq  # [B, T, 256]

            # Slot 2 (motion): temporal or legacy motion
            mot_slot = temporal_seq  # [B, T, 256] (new scheme, always 256)

            # Per-frame fusion
            frame_features = []
            for t in range(T):
                vis = vis_slot[:, t, :]
                spa = spa_slot[:, t, :]
                mot = mot_slot[:, t, :]

                # Legacy motion encoder (per-frame, for old checkpoints)
                if use_motion and motion_model and t > 0:
                    kd = kp[:, t, :, :2] - kp[:, t - 1, :, :2]
                    fh = torch.zeros(B, 128, device=device)
                    fh[:, :21] = torch.norm(kd, dim=2)
                    kpd = torch.cat([kd.reshape(B, 42), torch.zeros(B, 21, device=device)], dim=1)
                    mot = motion_model(fh, kpd)  # [B, 128]

                fused = fusion_model(vis, spa, mot)
                frame_features.append(fused.unsqueeze(1))

            fused_seq = torch.cat(frame_features, dim=1)

            logits = tcn_model(fused_seq).mean(dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.cpu().numpy())

    # ── Metrics ──
    from collections import Counter
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    acc = correct / len(all_labels)

    cls_correct = Counter()
    cls_total = Counter()
    for p, l in zip(all_preds, all_labels):
        cls_total[l] += 1
        if p == l:
            cls_correct[l] += 1

    print(f'\n=== DL Evaluation Results ===')
    print(f'Accuracy: {acc:.4f} ({acc*100:.1f}%)')
    print(f'\nPer-class:')
    for lid in sorted(cls_total.keys()):
        name = idx_to_label.get(lid, str(lid))
        c = cls_correct.get(lid, 0)
        t = cls_total[lid]
        print(f'  {name:15s}: {c}/{t} ({c/t*100:.1f}%)' if t else f'  {name:15s}: 0/0')

    # ── Confusion matrix ──
    conf = Counter()
    for p, l in zip(all_preds, all_labels):
        conf[(l, p)] += 1

    print(f'\nConfusion matrix (rows=true, cols=pred):')
    header = ' ' * 15 + ''.join(f'{idx_to_label.get(i, str(i)):>8s}' for i in range(num_classes))
    print(header)
    for i in range(num_classes):
        row = f'{idx_to_label.get(i, str(i)):15s}'
        for j in range(num_classes):
            row += f'{conf.get((i, j), 0):8d}'
        print(row)

    # ── Save ──
    result = {
        'accuracy': acc,
        'num_samples': len(all_labels),
        'num_classes': num_classes,
        'per_class': {
            idx_to_label.get(lid, str(lid)): {
                'correct': cls_correct.get(lid, 0),
                'total': cls_total[lid],
            }
            for lid in sorted(cls_total.keys())
        },
        'predictions': [int(p) for p in all_preds],
        'labels': [int(l) for l in all_labels],
    }
    out_path = args.data.replace('.json', '_dl_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {out_path}')

    if args.output_confusion:
        csv_path = args.data.replace('.json', '_confusion.csv')
        with open(csv_path, 'w') as f:
            f.write(',' + ','.join(idx_to_label.get(i, str(i)) for i in range(num_classes)) + '\n')
            for i in range(num_classes):
                f.write(idx_to_label.get(i, str(i)))
                for j in range(num_classes):
                    f.write(f',{conf.get((i, j), 0)}')
                f.write('\n')
        print(f'Confusion CSV: {csv_path}')


if __name__ == '__main__':
    main()
