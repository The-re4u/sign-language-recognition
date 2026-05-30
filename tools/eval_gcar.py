# coding:utf-8
"""Evaluate GCAR baseline on test sets."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_utils import SignLanguageDataset, collate_fn

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--data_dir', default='data')
p.add_argument('--gcar_ckpt', required=True, help='GCAR best_model.pth')
p.add_argument('--feat_ckpt', required=True, help='Feature extractor checkpoint (E4)')
p.add_argument('--batch_size', type=int, default=8)
p.add_argument('--geometric', action='store_true', help='GCAR+Geo variant')
args = p.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load feature extractors (frozen, from E4 checkpoint) ──
from core.feature.spatial_gcn import SpatialGCN
from core.feature.hand_shape_context import HandShapeContext
from core.feature.visual_encoder import LightweightVisualEncoder
from core.feature.multimodal_fusion import CrossModalFusion, GatedFusion

ckpt = torch.load(args.feat_ckpt, map_location=device, weights_only=False)
ckpt_mods = ckpt.get('modalities', {})
use_spatial = ckpt_mods.get('spatial', True)
use_cnn = ckpt_mods.get('cnn', True)
use_geo = ckpt_mods.get('geometric', False)

# Auto angle_dim
angle_dim = 64
if 'spatial_model' in ckpt:
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

gated_fusion = None
if use_cnn and use_geo:
    gated_fusion = GatedFusion().to(device).eval()
    if 'gated_fusion' in ckpt:
        gated_fusion.load_state_dict(ckpt['gated_fusion'])

vis_dim = 512 if use_cnn else 256
fusion = CrossModalFusion(visual_dim=vis_dim, motion_dim=256).to(device).eval()
fusion.load_state_dict(ckpt['fusion_model'])

# ── Load GCAR ──
from core.temporal.gcar import GCAR
geo_proj = None
if args.geometric:
    geo_proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU()).to(device).eval()

gcar = GCAR(input_dim=256, num_classes=10).to(device).eval()
gckpt = torch.load(args.gcar_ckpt, map_location=device, weights_only=False)
gcar.load_state_dict(gckpt['gcar_model'])
if geo_proj and 'geo_proj' in gckpt:
    geo_proj.load_state_dict(gckpt['geo_proj'])
print(f'Loaded GCAR (val_acc={gckpt.get("val_acc", "?")}) {"+Geo" if args.geometric else ""}')

# ── Data ──
dataset = SignLanguageDataset(args.data, data_dir=args.data_dir, use_cnn=use_cnn)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}

# ── Evaluate ──
all_preds, all_labels = [], []
with torch.no_grad():
    for kp, roi, flow, lbl, ln in loader:
        kp, roi, lbl = kp.to(device), roi.to(device), lbl.to(device)
        B, T = kp.shape[:2]
        kp_f = kp.reshape(B * T, 21, 3)
        roi_f = roi.reshape(B * T, 3, 96, 96)

        spa_seq = spatial(kp_f).reshape(B, T, 256)
        geo_seq = geometric(kp) if geometric else torch.zeros(B, T, 256, device=device)
        cnn_seq = cnn(roi_f).reshape(B, T, 512) if cnn else torch.zeros(B, T, 512, device=device)

        if gated_fusion:
            vis = gated_fusion(cnn_seq.reshape(B*T, 512), geo_seq.reshape(B*T, 256)).reshape(B, T, 512)
        elif use_cnn:
            vis = cnn_seq
        elif use_geo:
            vis = geo_seq
        else:
            vis = torch.zeros(B, T, vis_dim, device=device)

        feats = []
        for t_idx in range(T):
            f = fusion(vis[:, t_idx, :], spa_seq[:, t_idx, :], torch.zeros(B, 256, device=device))
            feats.append(f.unsqueeze(1))
        fused_seq = torch.cat(feats, dim=1)

        if geo_proj:
            geo_seq = geometric(kp) if geometric else torch.zeros(B, T, 256, device=device)
            fused_seq = geo_proj(torch.cat([fused_seq, geo_seq], dim=-1))
        logits = gcar(fused_seq).mean(dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds); all_labels.extend(lbl.cpu().numpy())

acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
print(f'\nAccuracy: {acc:.4f} ({acc*100:.1f}%)')

from collections import Counter
cls_c = Counter(); cls_t = Counter()
for p, l in zip(all_preds, all_labels):
    cls_t[l] += 1
    if p == l: cls_c[l] += 1

for lid in sorted(cls_t.keys()):
    name = idx_to_label.get(lid, str(lid))
    print(f'  {name:15s}: {cls_c.get(lid,0)}/{cls_t[lid]} ({cls_c.get(lid,0)/cls_t[lid]*100:.1f}%)' if cls_t[lid] else '')

# Save
result = {
    'accuracy': acc, 'num_samples': len(all_labels), 'num_classes': 10,
    'per_class': {idx_to_label.get(lid, str(lid)): {'correct': cls_c.get(lid,0), 'total': cls_t[lid]} for lid in sorted(cls_t.keys())},
    'predictions': [int(p) for p in all_preds], 'labels': [int(l) for l in all_labels],
}
out_path = args.data.replace('.json', '_gcar_result.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f'Saved: {out_path}')
