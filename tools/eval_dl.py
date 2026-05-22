# coding:utf-8
"""DL model evaluation on test set — outputs metrics matching validate.py format."""
import sys, os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.feature.spatial_gcn import SpatialGCN
from core.feature.motion_encoder import MotionEncoder
from core.feature.multimodal_fusion import CrossModalFusion
from core.feature.visual_encoder import LightweightVisualEncoder
from core.temporal.slowfast_tcn import SlowFastTCN
from utils.data_utils import SignLanguageDataset, collate_fn

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/test.json')
parser.add_argument('--data_dir', default='data/my_sequences')
parser.add_argument('--checkpoint', default='models/checkpoints/checkpoint_epoch_50.pth')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--no_visual', action='store_true')
parser.add_argument('--no_motion', action='store_true')
parser.add_argument('--no_gcn', action='store_true')
args = parser.parse_args()

device = torch.device('cpu')

# Load data
dataset = SignLanguageDataset(args.data, data_dir=args.data_dir)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
labels_list = sorted(set(a.get('label', a.get('gesture', 'unknown'))
                         for a in dataset.annotations))
label_to_idx = dataset.label_to_idx
idx_to_label = {v: k for k, v in label_to_idx.items()}
num_classes = len(label_to_idx)
print(f'Classes: {num_classes}, samples: {len(dataset)}')

# Load model
spatial = SpatialGCN().to(device).eval()
motion = MotionEncoder().to(device).eval()
fusion = CrossModalFusion().to(device).eval()
tcn = SlowFastTCN(input_dim=256, num_classes=14).to(device).eval()  # 14 = checkpoint class count
visual = LightweightVisualEncoder().to(device).eval()

ckpt = torch.load(args.checkpoint, map_location=device)
spatial.load_state_dict(ckpt['spatial_model'])
motion.load_state_dict(ckpt['motion_model'])
fusion.load_state_dict(ckpt['fusion_model'])
tcn.load_state_dict(ckpt['tcn_model'])
if 'visual_model' in ckpt:
    visual.load_state_dict(ckpt['visual_model'])
print(f'Loaded: {args.checkpoint}')

# Evaluate
all_preds, all_labels = [], []
with torch.no_grad():
    for kp, roi, lbl, ln in loader:
        kp, roi, lbl = kp.to(device), roi.to(device), lbl.to(device)
        B, T = kp.shape[:2]

        # GCN
        spatial_seq = torch.zeros(B, T, 256, device=device)
        if not args.no_gcn:
            kp_flat = kp[:, :, :, :3].reshape(B * T, 21, 3)
            spatial_seq = spatial(kp_flat).reshape(B, T, 256)

        # Visual
        visual_seq = torch.zeros(B, T, 512, device=device)
        if not args.no_visual:
            roi_flat = roi.reshape(B * T, 3, 96, 96)
            visual_seq = visual(roi_flat).reshape(B, T, 512)

        # Motion + fusion per-frame
        fused_feats = []
        for t in range(T):
            sp = spatial_seq[:, t, :]
            vi = visual_seq[:, t, :]
            mo = torch.zeros(B, 128, device=device)
            if not args.no_motion and t > 0:
                kd = kp[:, t, :, :2] - kp[:, t-1, :, :2]
                fh = torch.zeros(B, 128, device=device)
                fh[:, :21] = torch.norm(kd, dim=2)
                kpd = torch.cat([kd.reshape(B, 42), torch.zeros(B, 21, device=device)], dim=1)
                mo = motion(fh, kpd)
            fused_feats.append(fusion(vi, sp, mo).unsqueeze(1))
        fused_seq = torch.cat(fused_feats, dim=1)

        logits = tcn(fused_seq).mean(dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbl.cpu().numpy())

# Metrics
from collections import Counter
correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
acc = correct / len(all_labels)

# Per-class
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

# Confusion matrix
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

# Save
result = {
    'accuracy': acc,
    'num_samples': len(all_labels),
    'num_classes': num_classes,
    'per_class': {idx_to_label.get(lid, str(lid)): {'correct': cls_correct.get(lid, 0), 'total': cls_total[lid]}
                  for lid in sorted(cls_total.keys())},
    'predictions': [int(p) for p in all_preds],
    'labels': [int(l) for l in all_labels],
}
out_path = args.data.replace('.json', '_dl_result.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f'\nSaved: {out_path}')
