# coding:utf-8
"""
Training script for sign language recognition model v2.0.

Supports:
- Multi-modal feature extraction (spatial GCN, motion from keypoint deltas)
- Visual encoder when ROI data is available
- Cross-modal fusion (Transformer)
- SlowFast-TCN temporal modeling
- CE loss with temporal mean pooling (sequence → single label)
- Mixed precision (AMP), cosine annealing with warmup
- Checkpoint saving every N epochs

Usage:
  python train.py --data data/annotations.json --epochs 50
  python train.py --data data/annotations.json --val_data data/annotations_val.json --epochs 100 --batch_size 8
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to model config YAML')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data annotation JSON')
    parser.add_argument('--data_dir', type=str, default='data/sequences',
                        help='Directory containing .npy sequence files')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data annotation JSON')
    parser.add_argument('--output', type=str, default='models/checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--use_visual', action='store_true',
                        help='(deprecated) Use CNN visual encoder — prefer --cnn')
    parser.add_argument('--no_visual', action='store_true',
                        help='(deprecated) Disable visual — now default')
    parser.add_argument('--no_motion', action='store_true',
                        help='(deprecated) Disable motion — now default')
    # New modality flags (v6.0) — CNN is DEFAULT ON for task book compliance
    parser.add_argument('--no_spatial', action='store_true',
                        help='Disable SpatialGCN (ablation)')
    parser.add_argument('--no_cnn', action='store_true',
                        help='Disable CNN visual encoder (ablation)')
    parser.add_argument('--geometric', action='store_true',
                        help='Enable HandShapeContext dual-stream geometric encoder')
    parser.add_argument('--temporal', action='store_true',
                        help='Enable TemporalEncoder with per-keypoint attention')
    parser.add_argument('--bimanual_attn', action='store_true',
                        help='Enable BimanualCrossAttention for dual-hand data')
    parser.add_argument('--part_mixing', action='store_true',
                        help='Enable SKIM Part Mixing augmentation')
    parser.add_argument('--part_mix_prob', type=float, default=0.3,
                        help='Part Mixing application probability (default: 0.3)')
    parser.add_argument('--semantic', action='store_true',
                        help='Enable semantic modality (contrastive learning with class prototypes)')
    parser.add_argument('--semantic_lambda', type=float, default=0.1,
                        help='Weight for semantic contrastive loss (default: 0.1)')
    parser.add_argument('--semantic_dim', type=int, default=256,
                        help='Semantic embedding dimension (default: 256)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable online keypoint + ROI augmentation')
    parser.add_argument('--unfreeze_cnn', type=int, default=0,
                        help='Unfreeze last N MobileNetV3 blocks (0=all frozen, 3 recommended)')
    parser.add_argument('--angle_dim', type=int, default=64,
                        help='SpatialGCN angle encoder hidden dim (default: 64, E1: 128)')
    parser.add_argument('--class_weight', action='store_true',
                        help='Use WeightedRandomSampler for class balance')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override dropout rate (default: 0.1)')
    parser.add_argument('--tcn_kernel', type=int, default=None,
                        help='Override TCN slow-path kernel size (default: 7)')
    parser.add_argument('--gcn_layers', type=int, default=None,
                        help='Override GCN layer count (2-4, default: 3)')
    parser.add_argument('--warmup_epochs', type=int, default=None,
                        help='Override LR warmup epochs (default: 5)')
    return parser.parse_args()


def train():
    args = parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.cuda.amp import GradScaler, autocast
    except ImportError:
        print('PyTorch is required. Install: pip install torch torchvision')
        sys.exit(1)

    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False

    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = not args.no_amp and device.type == 'cuda'
    print(f'Device: {device}, AMP: {use_amp}')

    # ---- Build models ----
    print('Building models...')
    from core.feature.spatial_gcn import SpatialGCN
    from core.feature.multimodal_fusion import CrossModalFusion
    from core.temporal.slowfast_tcn import SlowFastTCN

    # Determine active modalities (CNN default ON for task book compliance)
    use_spatial = not args.no_spatial
    use_geometric = args.geometric
    use_temporal = args.temporal
    use_cnn = not args.no_cnn  # CNN is default ON
    use_bimanual = args.bimanual_attn
    use_part_mixing = args.part_mixing

    # Require at least one modality
    if not any([use_spatial, use_geometric, use_temporal, use_cnn]):
        print('ERROR: At least one modality must be enabled')
        sys.exit(1)

    # Spatial encoder (E1: --angle_dim 128 for stronger finger modeling)
    spatial_model = SpatialGCN(angle_dim=args.angle_dim).to(device) if use_spatial else None

    # Geometric encoder (HandShapeContext — replaces CNN visual)
    geometric_model = None
    if use_geometric:
        from core.feature.hand_shape_context import HandShapeContext
        geometric_model = HandShapeContext().to(device)

    # Temporal encoder (trajectory — replaces optical flow)
    temporal_model = None
    if use_temporal:
        from core.feature.temporal_encoder import TemporalEncoder
        temporal_model = TemporalEncoder().to(device)

    # CNN visual encoder (E2: --unfreeze_cnn 3 for partial backbone fine-tuning)
    cnn_visual_model = None
    if use_cnn:
        from core.feature.visual_encoder import LightweightVisualEncoder
        cnn_visual_model = LightweightVisualEncoder(
            freeze_backbone=(args.unfreeze_cnn == 0),
            unfreeze_blocks=args.unfreeze_cnn).to(device)

    # CrossModalFusion: slot dimensions depend on which encoders are active
    vis_dim = 256 if not use_cnn else 512
    mot_dim = 256  # always 256 in new scheme
    fusion_model = CrossModalFusion(visual_dim=vis_dim, motion_dim=mot_dim).to(device)

    # BimanualCrossAttention (Park et al., TPAMI 2025 [24])
    bimanual_model = None
    if use_bimanual:
        from core.feature.bimanual_attention import BimanualCrossAttention
        bimanual_model = BimanualCrossAttention(dim=256).to(device)

    # When CNN + geometric both active: gated fusion in visual slot
    gated_fusion = None
    if use_cnn and use_geometric:
        from core.feature.multimodal_fusion import GatedFusion
        gated_fusion = GatedFusion().to(device)

    # Semantic modality (contrastive learning with class prototypes)
    semantic_projector = None
    use_semantic = args.semantic
    if use_semantic:
        from core.feature.semantic_encoder import SemanticProjector, semantic_contrastive_loss
        semantic_projector = SemanticProjector(
            feat_dim=256, semantic_dim=args.semantic_dim,
            num_classes=10).to(device)

    # Apply hparam overrides for sensitivity sweep
    tcn_kernel = args.tcn_kernel if args.tcn_kernel is not None else 7
    tcn_dropout = args.dropout if args.dropout is not None else 0.1
    tcn_model = SlowFastTCN(input_dim=256, num_classes=10,
                            slow_kernel=tcn_kernel, dropout=tcn_dropout).to(device)

    # Collect trainable parameters
    params = list(fusion_model.parameters()) + list(tcn_model.parameters())
    if spatial_model:
        params += list(spatial_model.parameters())
    if geometric_model:
        params += list(geometric_model.parameters())
    if temporal_model:
        params += list(temporal_model.parameters())
    if cnn_visual_model:
        params += list(cnn_visual_model.parameters())
    if bimanual_model:
        params += list(bimanual_model.parameters())
    if gated_fusion:
        params += list(gated_fusion.parameters())
    if semantic_projector:
        params += list(semantic_projector.parameters())

    total_params = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in params if p.requires_grad)
    parts = []
    if use_spatial: parts.append('spatial')
    if use_geometric: parts.append('geo')
    if use_temporal: parts.append('temp')
    if use_cnn: parts.append('cnn')
    if use_bimanual: parts.append('biman')
    print(f'Modalities: {"+".join(parts)}')
    print(f'Parameters: {total_params:,} total ({trainable:,} trainable)')

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(params, lr=args.lr, weight_decay=1e-4)

    # Cosine annealing with linear warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
    warmup_epochs = 5
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)

    def warmup_fn(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, warmup_fn)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if spatial_model:
            spatial_model.load_state_dict(checkpoint['spatial_model'])
        fusion_model.load_state_dict(checkpoint['fusion_model'])
        tcn_model.load_state_dict(checkpoint['tcn_model'])
        if geometric_model and 'geometric_model' in checkpoint:
            geometric_model.load_state_dict(checkpoint['geometric_model'])
        if temporal_model and 'temporal_model' in checkpoint:
            temporal_model.load_state_dict(checkpoint['temporal_model'])
        if cnn_visual_model and 'visual_model' in checkpoint:
            cnn_visual_model.load_state_dict(checkpoint['visual_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}')

    scaler = GradScaler(device_type=device.type) if use_amp else None

    # ---- Data ----
    from utils.data_utils import create_data_loader
    train_loader = create_data_loader(args.data, batch_size=args.batch_size,
                                      data_dir=args.data_dir,
                                      class_weight=args.class_weight,
                                      augment=args.augment, use_cnn=use_cnn,
                                      part_mixing=use_part_mixing,
                                      part_mix_prob=args.part_mix_prob)
    val_loader = None
    if args.val_data:
        val_loader = create_data_loader(args.val_data, batch_size=args.batch_size,
                                        data_dir=args.data_dir, shuffle=False,
                                        augment=False, use_cnn=use_cnn,
                                        part_mixing=False)

    # ---- Loss ----
    ce_loss_fn = nn.CrossEntropyLoss()

    print(f'\nTraining {args.epochs} epochs, warmup={warmup_epochs}, batches={len(train_loader)}')

    # Logging
    import json as _json
    loss_log_path = os.path.join(args.output, 'loss_log.json')
    metrics_log_path = os.path.join(args.output, 'training_metrics.json')
    loss_history = []
    metrics_history = []  # v3.4: save train/val loss + accuracy for thesis figures

    epoch_iter = tqdm(range(start_epoch, args.epochs), desc='Epochs', unit='ep') if HAS_TQDM else range(start_epoch, args.epochs)
    for epoch in epoch_iter:
        # ---- Training ----
        if spatial_model: spatial_model.train()
        if geometric_model: geometric_model.train()
        if temporal_model: temporal_model.train()
        if cnn_visual_model: cnn_visual_model.train()
        if bimanual_model: bimanual_model.train()
        if semantic_projector: semantic_projector.train()
        fusion_model.train()
        tcn_model.train()

        total_loss = 0.0

        batch_iter = enumerate(train_loader)
        if HAS_TQDM:
            batch_iter = tqdm(batch_iter, total=len(train_loader), desc=f'Epoch {epoch+1}', unit='batch', leave=False)
        for batch_idx, (sequences, rois, flow_hists, labels, lengths) in batch_iter:
            sequences = sequences.to(device)   # [B, T, 21, 3]
            rois = rois.to(device)              # [B, T, 3, 96, 96]
            labels = labels.to(device)          # [B]
            B, T, N, _ = sequences.shape

            optimizer.zero_grad()

            with autocast(device_type=device.type) if use_amp else nullcontext():
                kp_flat = sequences.reshape(B * T, 21, 3)

                # Spatial: vectorized GCN
                if use_spatial:
                    spatial_seq = spatial_model(kp_flat).reshape(B, T, 256)
                else:
                    spatial_seq = torch.zeros(B, T, 256, device=device)

                # Geometric: HandShapeContext (vectorized)
                if use_geometric:
                    geometric_seq = geometric_model(sequences)  # [B, T, 256]
                else:
                    geometric_seq = torch.zeros(B, T, 256, device=device)

                # Temporal: TrajectoryEncoder (per-sequence)
                if use_temporal:
                    temporal_seq = temporal_model(sequences)  # [B, T, 256]
                else:
                    temporal_seq = torch.zeros(B, T, 256, device=device)

                # CNN visual: MobileNetV3 (vectorized)
                if use_cnn:
                    roi_flat = rois.reshape(B * T, 3, 96, 96)
                    cnn_seq = cnn_visual_model(roi_flat).reshape(B, T, 512)
                else:
                    cnn_seq = torch.zeros(B, T, 512, device=device)

                # Fill CrossModalFusion slots
                # Slot 0 (visual): CNN(512) + gated fusion if both active
                if use_cnn and use_geometric and gated_fusion is not None:
                    cnn_flat = cnn_seq.reshape(B * T, 512)
                    geo_flat = geometric_seq.reshape(B * T, 256)
                    vis_slot = gated_fusion(cnn_flat, geo_flat).reshape(B, T, 512)
                elif use_cnn:
                    vis_slot = cnn_seq  # [B, T, 512]
                elif use_geometric:
                    vis_slot = geometric_seq  # [B, T, 256], vis_dim=256
                else:
                    vis_slot = torch.zeros(B, T, vis_dim, device=device)

                # Slot 1 (spatial): spatial (256) or zeros
                spa_slot = spatial_seq

                # Slot 2 (motion): temporal (256) or zeros
                mot_slot = temporal_seq

                # Per-frame fusion
                frame_features = []
                for t in range(T):
                    vis = vis_slot[:, t, :]
                    spa = spa_slot[:, t, :]
                    mot = mot_slot[:, t, :]

                    # Bimanual cross-attention (applied to spatial + temporal if both hands)
                    if bimanual_model and use_bimanual:
                        # For single-hand data (current), right=zeros → identity
                        spa_right = torch.zeros_like(spa)
                        mot_right = torch.zeros_like(mot)
                        spa, spa_right = bimanual_model(spa, spa_right)
                        mot, mot_right = bimanual_model(mot, mot_right)

                    fused = fusion_model(vis, spa, mot)
                    frame_features.append(fused.unsqueeze(1))

                fused_seq = torch.cat(frame_features, dim=1)

                # TCN → mean pooling → CE classification
                logits = tcn_model(fused_seq)  # [B, T, num_classes]
                mean_logits = logits.mean(dim=1)  # [B, C=10]
                total = ce_loss_fn(mean_logits, labels)

                # Semantic contrastive loss (per-frame or pooled)
                if semantic_projector:
                    pooled_feat = fused_seq.mean(dim=1)  # [B, 256]
                    proj = semantic_projector(pooled_feat)  # [B, D]
                    prototypes = semantic_projector.get_prototypes()  # [10, D]
                    sem_loss = semantic_contrastive_loss(proj, labels, prototypes)
                    total = total + args.semantic_lambda * sem_loss

            if scaler:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            total_loss += total.item()

        # Scheduler step
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        avg_loss = total_loss / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        loss_history.append({'epoch': epoch + 1, 'loss': avg_loss, 'lr': lr})
        with open(loss_log_path, 'w') as f:
            _json.dump(loss_history, f)
        status = f'Epoch {epoch+1}/{args.epochs} Loss: {avg_loss:.4f} LR: {lr:.2e}'
        if HAS_TQDM:
            epoch_iter.set_postfix_str(status)
        else:
            print(status)

        # Track best val model
        if not hasattr(train, '_best_val_acc'):
            train._best_val_acc = 0.0
            train._best_epoch = 0

        # Validation
        if val_loader:
            val_loss, val_acc = evaluate(
                spatial_model, geometric_model, temporal_model,
                cnn_visual_model, fusion_model, tcn_model,
                val_loader, device, ce_loss_fn,
                use_spatial=use_spatial, use_geometric=use_geometric,
                use_temporal=use_temporal, use_cnn=use_cnn,
                gated_fusion=gated_fusion,
                semantic_projector=semantic_projector,
                use_semantic=use_semantic,
                semantic_lambda=args.semantic_lambda)
            print(f'  Val Loss: {val_loss:.4f}  Acc: {val_acc:.2%}')

            # Save best model based on val_acc
            if val_acc > train._best_val_acc:
                train._best_val_acc = val_acc
                train._best_epoch = epoch + 1
                best_ckpt = {
                    'epoch': epoch + 1,
                    'fusion_model': fusion_model.state_dict(),
                    'tcn_model': tcn_model.state_dict(),
                    'val_acc': val_acc,
                    'modalities': {
                        'spatial': use_spatial,
                        'geometric': use_geometric,
                        'temporal': use_temporal,
                        'cnn': use_cnn,
                        'bimanual': use_bimanual,
                        'semantic': use_semantic,
                    },
                }
                if spatial_model:
                    best_ckpt['spatial_model'] = spatial_model.state_dict()
                if geometric_model:
                    best_ckpt['geometric_model'] = geometric_model.state_dict()
                if temporal_model:
                    best_ckpt['temporal_model'] = temporal_model.state_dict()
                if cnn_visual_model:
                    best_ckpt['visual_model'] = cnn_visual_model.state_dict()
                if bimanual_model:
                    best_ckpt['bimanual_model'] = bimanual_model.state_dict()
                if gated_fusion is not None:
                    best_ckpt['gated_fusion'] = gated_fusion.state_dict()
                if semantic_projector is not None:
                    best_ckpt['semantic_projector'] = semantic_projector.state_dict()
                torch.save(best_ckpt, os.path.join(args.output, 'best_model.pth'))
                print(f'  >> Best model saved (Val Acc={val_acc:.2%})')

            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': lr,
            })
            with open(metrics_log_path, 'w') as f:
                _json.dump(metrics_history, f, indent=2)

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            ckpt = {
                'epoch': epoch + 1,
                'fusion_model': fusion_model.state_dict(),
                'tcn_model': tcn_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }
            if spatial_model:
                ckpt['spatial_model'] = spatial_model.state_dict()
            if geometric_model:
                ckpt['geometric_model'] = geometric_model.state_dict()
            if temporal_model:
                ckpt['temporal_model'] = temporal_model.state_dict()
            if cnn_visual_model:
                ckpt['visual_model'] = cnn_visual_model.state_dict()
            if bimanual_model:
                ckpt['bimanual_model'] = bimanual_model.state_dict()
            if gated_fusion is not None:
                ckpt['gated_fusion'] = gated_fusion.state_dict()
            if semantic_projector is not None:
                ckpt['semantic_projector'] = semantic_projector.state_dict()
            ckpt_path = os.path.join(args.output, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(ckpt, ckpt_path)
            print(f'  Saved: {ckpt_path}')

    # Save final model (best was already saved during training)
    final_path = os.path.join(args.output, 'final_model.pth')
    final_ckpt = {
        'epoch': args.epochs,
        'fusion_model': fusion_model.state_dict(),
        'tcn_model': tcn_model.state_dict(),
        'loss': avg_loss,
    }
    if spatial_model:
        final_ckpt['spatial_model'] = spatial_model.state_dict()
    if geometric_model:
        final_ckpt['geometric_model'] = geometric_model.state_dict()
    if temporal_model:
        final_ckpt['temporal_model'] = temporal_model.state_dict()
    if cnn_visual_model:
        final_ckpt['visual_model'] = cnn_visual_model.state_dict()
    if bimanual_model:
        final_ckpt['bimanual_model'] = bimanual_model.state_dict()
    if gated_fusion is not None:
        final_ckpt['gated_fusion'] = gated_fusion.state_dict()
    if semantic_projector is not None:
        final_ckpt['semantic_projector'] = semantic_projector.state_dict()
    torch.save(final_ckpt, final_path)
    print(f'\nFinal model saved: {final_path}')
    print('Training complete! Run export:')
    print(f'  python export_onnx.py --checkpoint {final_path} --output models/sign_recognizer.onnx')


class nullcontext:
    """Fallback for when autocast is disabled."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


def evaluate(spatial_model, geometric_model, temporal_model,
             cnn_visual_model, fusion_model, tcn_model,
             val_loader, device, ce_loss_fn,
             use_spatial=True, use_geometric=False,
             use_temporal=False, use_cnn=False,
             gated_fusion=None, semantic_projector=None,
             use_semantic=False, semantic_lambda=0.1):
    """Run validation loop with configurable modalities."""
    import torch
    if spatial_model: spatial_model.eval()
    if geometric_model: geometric_model.eval()
    if temporal_model: temporal_model.eval()
    if cnn_visual_model: cnn_visual_model.eval()
    if gated_fusion: gated_fusion.eval()
    if semantic_projector: semantic_projector.eval()
    fusion_model.eval()
    tcn_model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for sequences, rois, flow_hists, labels, lengths in val_loader:
            sequences = sequences.to(device)
            rois = rois.to(device)
            labels = labels.to(device)
            B, T, N, _ = sequences.shape

            kp_flat = sequences.reshape(B * T, 21, 3)

            if use_spatial and spatial_model:
                spatial_seq = spatial_model(kp_flat).reshape(B, T, 256)
            else:
                spatial_seq = torch.zeros(B, T, 256, device=device)

            if use_geometric and geometric_model:
                geometric_seq = geometric_model(sequences)
            else:
                geometric_seq = torch.zeros(B, T, 256, device=device)

            if use_temporal and temporal_model:
                temporal_seq = temporal_model(sequences)
            else:
                temporal_seq = torch.zeros(B, T, 256, device=device)

            if use_cnn and cnn_visual_model:
                roi_flat = rois.reshape(B * T, 3, 96, 96)
                cnn_seq = cnn_visual_model(roi_flat).reshape(B, T, 512)
            else:
                cnn_seq = torch.zeros(B, T, 512, device=device)

            vis_dim = 512 if use_cnn else 256
            # Fill fusion slots (matches training forward pass)
            if use_cnn and use_geometric and gated_fusion is not None:
                cnn_flat = cnn_seq.reshape(B * T, 512)
                geo_flat = geometric_seq.reshape(B * T, 256)
                vis_slot = gated_fusion(cnn_flat, geo_flat).reshape(B, T, 512)
            elif use_cnn:
                vis_slot = cnn_seq
            elif use_geometric:
                vis_slot = geometric_seq
            else:
                vis_slot = torch.zeros(B, T, vis_dim, device=device)

            frame_features = []
            for t in range(T):
                fused = fusion_model(vis_slot[:, t, :],
                                     spatial_seq[:, t, :],
                                     temporal_seq[:, t, :])
                frame_features.append(fused.unsqueeze(1))

            fused_seq = torch.cat(frame_features, dim=1)
            logits = tcn_model(fused_seq)
            mean_logits = logits.mean(dim=1)
            total_loss += ce_loss_fn(mean_logits, labels).item()
            preds = mean_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += B

    return total_loss / len(val_loader), correct / total_samples


if __name__ == '__main__':
    train()
