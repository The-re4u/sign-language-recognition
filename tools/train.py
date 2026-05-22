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
                        help='Use visual encoder (requires ROI data)')
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
    from core.feature.motion_encoder import MotionEncoder
    from core.feature.multimodal_fusion import CrossModalFusion
    from core.temporal.slowfast_tcn import SlowFastTCN

    spatial_model = SpatialGCN().to(device)
    motion_model = MotionEncoder().to(device)
    fusion_model = CrossModalFusion().to(device)
    tcn_model = SlowFastTCN(input_dim=256, num_classes=14).to(device)

    # Visual encoder is optional (needs ROI images)
    visual_model = None
    if args.use_visual:
        from core.feature.visual_encoder import LightweightVisualEncoder
        visual_model = LightweightVisualEncoder().to(device)

    params = (list(spatial_model.parameters()) +
              list(motion_model.parameters()) +
              list(fusion_model.parameters()) +
              list(tcn_model.parameters()))
    if visual_model:
        params += list(visual_model.parameters())

    total_params = sum(p.numel() for p in params)
    print(f'Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)')

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
        spatial_model.load_state_dict(checkpoint['spatial_model'])
        motion_model.load_state_dict(checkpoint['motion_model'])
        fusion_model.load_state_dict(checkpoint['fusion_model'])
        tcn_model.load_state_dict(checkpoint['tcn_model'])
        if visual_model and 'visual_model' in checkpoint:
            visual_model.load_state_dict(checkpoint['visual_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}')

    scaler = GradScaler() if use_amp else None

    # ---- Data ----
    from utils.data_utils import create_data_loader
    train_loader = create_data_loader(args.data, batch_size=args.batch_size, data_dir=args.data_dir)
    val_loader = None
    if args.val_data:
        val_loader = create_data_loader(args.val_data, batch_size=args.batch_size, data_dir=args.data_dir, shuffle=False)

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
        spatial_model.train()
        motion_model.train()
        fusion_model.train()
        tcn_model.train()
        if visual_model:
            visual_model.train()

        total_loss = 0.0

        batch_iter = enumerate(train_loader)
        if HAS_TQDM:
            batch_iter = tqdm(batch_iter, total=len(train_loader), desc=f'Epoch {epoch+1}', unit='batch', leave=False)
        for batch_idx, (sequences, rois, labels, lengths) in batch_iter:
            sequences = sequences.to(device)   # [B, T, 21, 3]
            rois = rois.to(device)              # [B, T, 3, 96, 96]
            labels = labels.to(device)          # [B]
            B, T, N, _ = sequences.shape

            optimizer.zero_grad()

            with autocast() if use_amp else nullcontext():
                # v3.4: Vectorize GCN (biggest bottleneck) — rest per-frame for stability
                kp_flat = sequences[:, :, :, :3].reshape(B * T, 21, 3)
                spatial_flat = spatial_model(kp_flat)
                spatial_seq = spatial_flat.reshape(B, T, 256)

                # Visual: batch all frames
                if visual_model:
                    roi_flat = rois[:, :, :, :, :].reshape(B * T, 3, 96, 96)
                    visual_flat = visual_model(roi_flat)
                    visual_seq = visual_flat.reshape(B, T, 512)
                else:
                    visual_seq = torch.zeros(B, T, 512, device=device)

                # Motion + Fusion: per-frame (needs temporal deltas, more stable)
                frame_features = []
                for t in range(T):
                    spatial = spatial_seq[:, t, :]
                    visual = visual_seq[:, t, :]
                    if t > 0:
                        kp_diff = sequences[:, t, :, :2] - sequences[:, t-1, :, :2]
                        flow_hist = torch.zeros(B, 128, device=device)
                        mag = torch.norm(kp_diff, dim=2)
                        flow_hist[:, :21] = mag
                        angles = torch.atan2(kp_diff[:, :, 1], kp_diff[:, :, 0])
                        for j in range(21):
                            bin_idx = ((angles[:, j] / (2 * 3.14159) + 0.5) * 42).long().clamp(0, 41)
                            flow_hist[torch.arange(B), 21 + bin_idx] += mag[:, j]
                        kp_delta = torch.cat([kp_diff.reshape(B, 42), torch.zeros(B, 21, device=device)], dim=1)
                        motion = motion_model(flow_hist, kp_delta)
                    else:
                        motion = torch.zeros(B, 128, device=device)
                    fused = fusion_model(visual, spatial, motion)
                    frame_features.append(fused.unsqueeze(1))

                fused_seq = torch.cat(frame_features, dim=1)

                # TCN → mean pooling → CE classification
                logits = tcn_model(fused_seq)  # [B, T, num_classes]
                mean_logits = logits.mean(dim=1)  # [B, C=14]
                total = ce_loss_fn(mean_logits, labels)

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

        # Validation
        if val_loader:
            val_loss, val_acc = evaluate(spatial_model, motion_model, fusion_model,
                                         tcn_model, visual_model, val_loader, device,
                                         ce_loss_fn)
            print(f'  Val Loss: {val_loss:.4f}  Acc: {val_acc:.2%}')
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
                'spatial_model': spatial_model.state_dict(),
                'motion_model': motion_model.state_dict(),
                'fusion_model': fusion_model.state_dict(),
                'tcn_model': tcn_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }
            if visual_model:
                ckpt['visual_model'] = visual_model.state_dict()
            ckpt_path = os.path.join(args.output, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(ckpt, ckpt_path)
            print(f'  Saved: {ckpt_path}')

    # Save final model
    final_path = os.path.join(args.output, 'best_model.pth')
    torch.save({
        'epoch': args.epochs,
        'spatial_model': spatial_model.state_dict(),
        'motion_model': motion_model.state_dict(),
        'fusion_model': fusion_model.state_dict(),
        'tcn_model': tcn_model.state_dict(),
        'loss': avg_loss,
    }, final_path)
    print(f'\nFinal model saved: {final_path}')
    print('Training complete! Run export:')
    print(f'  python export_onnx.py --checkpoint {final_path} --output models/sign_recognizer.onnx')


class nullcontext:
    """Fallback for when autocast is disabled."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


def evaluate(spatial_model, motion_model, fusion_model, tcn_model,
             visual_model, val_loader, device, ce_loss_fn):
    """Run validation loop."""
    import torch
    spatial_model.eval()
    motion_model.eval()
    fusion_model.eval()
    tcn_model.eval()
    if visual_model:
        visual_model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for sequences, rois, labels, lengths in val_loader:
            sequences = sequences.to(device)
            rois = rois.to(device)
            labels = labels.to(device)
            B, T, N, _ = sequences.shape

            # Vectorized GCN + Visual, per-frame motion + fusion (same as training)
            kp_flat = sequences[:, :, :, :3].reshape(B * T, 21, 3)
            spatial_flat = spatial_model(kp_flat)
            spatial_seq = spatial_flat.reshape(B, T, 256)

            if visual_model:
                roi_flat = rois[:, :, :, :, :].reshape(B * T, 3, 96, 96)
                visual_flat = visual_model(roi_flat)
                visual_seq = visual_flat.reshape(B, T, 512)
            else:
                visual_seq = torch.zeros(B, T, 512, device=device)

            frame_features = []
            for t in range(T):
                spatial = spatial_seq[:, t, :]
                visual = visual_seq[:, t, :]
                if t > 0:
                    kp_diff = sequences[:, t, :, :2] - sequences[:, t-1, :, :2]
                    flow_hist = torch.zeros(B, 128, device=device)
                    mag = torch.norm(kp_diff, dim=2)
                    flow_hist[:, :21] = mag
                    angles = torch.atan2(kp_diff[:, :, 1], kp_diff[:, :, 0])
                    for j in range(21):
                        bin_idx = ((angles[:, j] / (2 * 3.14159) + 0.5) * 42).long().clamp(0, 41)
                        flow_hist[torch.arange(B), 21 + bin_idx] += mag[:, j]
                    kp_delta = torch.cat([kp_diff.reshape(B, 42), torch.zeros(B, 21, device=device)], dim=1)
                    motion = motion_model(flow_hist, kp_delta)
                else:
                    motion = torch.zeros(B, 128, device=device)
                fused = fusion_model(visual, spatial, motion)
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
