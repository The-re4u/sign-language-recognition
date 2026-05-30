# coding:utf-8
"""GCAR baseline training v2.0 — supports v6.0 multimodal checkpoints.

Loads frozen feature extractors from a main model checkpoint, then trains
GCAR temporal model on the extracted 256-dim fused features.

Usage:
  python tools/train_baseline.py --data data/train.json --val_data data/val.json \
      --data_dir data --checkpoint models/checkpoints/E4_spatial_cnn_geo/best_model.pth \
      --use_visual --epochs 50 --output models/baseline_gcar
"""
import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--val_data', required=True)
    p.add_argument('--data_dir', default='data')
    p.add_argument('--checkpoint', required=True, help='Main model checkpoint (feature extractors)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--output', default='models/baseline_gcar')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--geometric', action='store_true',
                   help='Concat HandShapeContext features to GCAR input (GCAR+Geo variant)')
    return p.parse_args()


def train():
    args = parse_args()
    import torch, torch.nn as nn
    from torch.optim import AdamW
    from torch.cuda.amp import GradScaler, autocast

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = not args.no_amp and device.type == 'cuda'
    print(f'Device: {device}')

    # ── Load checkpoint & auto-detect modalities ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_mods = ckpt.get('modalities', {})
    use_spatial = ckpt_mods.get('spatial', 'spatial_model' in ckpt)
    use_cnn = ckpt_mods.get('cnn', 'visual_model' in ckpt)
    use_geometric = ckpt_mods.get('geometric', 'geometric_model' in ckpt)
    use_gated = use_cnn and use_geometric and ('gated_fusion' in ckpt)
    use_geo_gcar = args.geometric  # GCAR+Geo variant
    print(f'Modalities: spatial={use_spatial} cnn={use_cnn} geo={use_geometric} gated={use_gated} gcar_geo={use_geo_gcar}')

    # ── Build feature extractors ──
    from core.feature.spatial_gcn import SpatialGCN
    from core.feature.multimodal_fusion import CrossModalFusion

    # Auto-detect angle_dim
    angle_dim = 64
    if use_spatial and 'spatial_model' in ckpt:
        w = ckpt['spatial_model'].get('final_fusion.0.weight')
        if w is not None:
            angle_dim = w.shape[1] - 256

    spatial = SpatialGCN(angle_dim=angle_dim).to(device).eval()
    spatial.load_state_dict(ckpt['spatial_model'])
    for p in spatial.parameters():
        p.requires_grad = False

    geometric = None
    if use_geometric and 'geometric_model' in ckpt:
        from core.feature.hand_shape_context import HandShapeContext
        geometric = HandShapeContext().to(device).eval()
        geometric.load_state_dict(ckpt['geometric_model'])
        for p in geometric.parameters():
            p.requires_grad = False

    cnn_visual = None
    if use_cnn and 'visual_model' in ckpt:
        from core.feature.visual_encoder import LightweightVisualEncoder
        cnn_visual = LightweightVisualEncoder(freeze_backbone=True).to(device).eval()
        cnn_visual.load_state_dict(ckpt['visual_model'])
        for p in cnn_visual.parameters():
            p.requires_grad = False

    gated_fusion = None
    if use_gated:
        from core.feature.multimodal_fusion import GatedFusion
        gated_fusion = GatedFusion().to(device).eval()
        gated_fusion.load_state_dict(ckpt['gated_fusion'])
        for p in gated_fusion.parameters():
            p.requires_grad = False

    vis_dim = 512 if use_cnn else 256
    fusion = CrossModalFusion(visual_dim=vis_dim, motion_dim=256).to(device).eval()
    fusion.load_state_dict(ckpt['fusion_model'])
    for p in fusion.parameters():
        p.requires_grad = False

    # ── Build GCAR ──
    from core.temporal.gcar import GCAR
    geo_proj = None
    gcar_input_dim = 256
    if use_geo_gcar:
        geo_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        ).to(device)
        gcar_input_dim = 256  # after projection from 512→256
    gcar = GCAR(input_dim=gcar_input_dim, num_classes=10).to(device)
    n_p = sum(p.numel() for p in gcar.parameters())
    if geo_proj:
        n_p += sum(p.numel() for p in geo_proj.parameters())
    print(f'GCAR(+Geo) params: {n_p:,} ({n_p/1e6:.2f}M)')
    print(f'Feature extractors: FROZEN')

    optimizer = AdamW(list(gcar.parameters()) + (list(geo_proj.parameters()) if geo_proj else []), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() if use_amp else None
    ce_loss = nn.CrossEntropyLoss()

    # ── Data ──
    from utils.data_utils import create_data_loader
    train_loader = create_data_loader(args.data, batch_size=args.batch_size,
                                      data_dir=args.data_dir, augment=True, use_cnn=use_cnn)
    val_loader = create_data_loader(args.val_data, batch_size=args.batch_size,
                                    data_dir=args.data_dir, shuffle=False, use_cnn=use_cnn)

    # ── Scheduler ──
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
    warmup_epochs = 5
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    warmup = LambdaLR(optimizer, lambda e: (e + 1) / warmup_epochs if e < warmup_epochs else 1.0)

    metrics_log = []
    best_val_acc = 0.0

    class nullctx:
        def __enter__(self): return None
        def __exit__(self, *a): pass

    for epoch in range(args.epochs):
        gcar.train()
        total_loss = 0.0

        for kp, roi, flow, labels, lengths in train_loader:
            kp, roi, labels = kp.to(device), roi.to(device), labels.to(device)
            B, T = kp.shape[:2]

            optimizer.zero_grad()
            with autocast() if use_amp else nullctx():
                with torch.no_grad():
                    kp_f = kp.reshape(B * T, 21, 3)
                    roi_f = roi.reshape(B * T, 3, 96, 96)

                    spa_seq = spatial(kp_f).reshape(B, T, 256) if use_spatial else torch.zeros(B, T, 256, device=device)
                    geo_seq = geometric(kp).to(device) if geometric else torch.zeros(B, T, 256, device=device)
                    cnn_seq = cnn_visual(roi_f).reshape(B, T, 512) if cnn_visual else torch.zeros(B, T, 512, device=device)

                    if gated_fusion:
                        vis = gated_fusion(cnn_seq.reshape(B*T, 512), geo_seq.reshape(B*T, 256)).reshape(B, T, 512)
                    elif use_cnn:
                        vis = cnn_seq
                    elif use_geometric:
                        vis = geo_seq
                    else:
                        vis = torch.zeros(B, T, vis_dim, device=device)

                    feats = []
                    for t_idx in range(T):
                        f = fusion(vis[:, t_idx, :], spa_seq[:, t_idx, :],
                                   torch.zeros(B, 256, device=device))
                        feats.append(f.unsqueeze(1))
                    fused_seq = torch.cat(feats, dim=1)

                # GCAR+Geo: concat geometric features
                if geo_proj:
                    geo_seq = geometric(kp).to(device) if geometric else torch.zeros(B, T, 256, device=device)
                    fused_seq = geo_proj(torch.cat([fused_seq, geo_seq], dim=-1))

                logits = gcar(fused_seq)
                loss = ce_loss(logits.mean(dim=1), labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(gcar.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gcar.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()

        if epoch < warmup_epochs: warmup.step()
        else: cosine.step()

        avg_loss = total_loss / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{args.epochs} Loss: {avg_loss:.4f} LR: {lr:.2e}')

        # Validation
        if val_loader:
            gcar.eval()
            v_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for kp, roi, flow, labels, lengths in val_loader:
                    kp, roi, labels = kp.to(device), roi.to(device), labels.to(device)
                    B, T = kp.shape[:2]
                    kp_f = kp.reshape(B * T, 21, 3)
                    roi_f = roi.reshape(B * T, 3, 96, 96)

                    spa_seq = spatial(kp_f).reshape(B, T, 256) if use_spatial else torch.zeros(B, T, 256, device=device)
                    geo_seq = geometric(kp).to(device) if geometric else torch.zeros(B, T, 256, device=device)
                    cnn_seq = cnn_visual(roi_f).reshape(B, T, 512) if cnn_visual else torch.zeros(B, T, 512, device=device)

                    if gated_fusion:
                        vis = gated_fusion(cnn_seq.reshape(B*T, 512), geo_seq.reshape(B*T, 256)).reshape(B, T, 512)
                    elif use_cnn:
                        vis = cnn_seq
                    elif use_geometric:
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

                    logits = gcar(fused_seq)
                    v_loss += ce_loss(logits.mean(dim=1), labels).item()
                    preds = logits.mean(dim=1).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += B

            val_acc = correct / total
            val_l = v_loss / len(val_loader)
            print(f'  Val Loss: {val_l:.4f}  Acc: {val_acc:.2%}')
            metrics_log.append({
                'epoch': epoch + 1, 'train_loss': avg_loss,
                'val_loss': val_l, 'val_acc': val_acc, 'lr': lr,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                sd = {'epoch': epoch + 1, 'gcar_model': gcar.state_dict(), 'val_acc': val_acc}
                if geo_proj: sd['geo_proj'] = geo_proj.state_dict()
                torch.save(sd, os.path.join(args.output, 'best_model.pth'))

    with open(os.path.join(args.output, 'training_metrics.json'), 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f'\nBest val_acc: {best_val_acc:.2%}')
    print(f'Output: {args.output}')


if __name__ == '__main__':
    train()
