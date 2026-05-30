# coding:utf-8
"""Lightweight visual encoder for hand ROI appearance features.

Uses MobileNetV3-Small backbone (pretrained), frozen during training,
with a small trainable projection head. Designed for 96×96 hand ROIs.

Backbone: torchvision mobilenet_v3_small (2.5M params, frozen)
Projection: 2-layer MLP (~0.15M params, trainable)
Total trainable: ~0.15M
"""
import torch
import torch.nn as nn

try:
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class LightweightVisualEncoder(nn.Module):
    """MobileNetV3-Small backbone + learnable projection for hand ROI encoding.

    Input:  [B, 3, 96, 96]  RGB hand ROI
    Output: [B, 512]         visual appearance embedding
    Trainable params: ~0.15M (backbone frozen)
    """

    def __init__(self, in_channels=3, roi_size=(96, 96), out_features=512,
                 freeze_backbone=True, unfreeze_blocks=0):
        super().__init__()
        self.roi_size = roi_size
        self.out_features = out_features

        if HAS_TORCHVISION:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v3_small(weights=weights)
            backbone_dim = 576  # MobileNetV3-Small feature dim
        else:
            # Fallback: tiny CNN when torchvision unavailable
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            backbone_dim = 128

        # Freeze backbone (with optional partial unfreeze)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_blocks > 0 and HAS_TORCHVISION:
                # Unfreeze last N feature blocks (MobileNetV3-Small has 13 layers in features)
                features = list(self.backbone.features.children())
                n_features = len(features)
                for i in range(max(0, n_features - unfreeze_blocks), n_features):
                    for p in features[i].parameters():
                        p.requires_grad = True

        # Trainable projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, out_features),
        )

    def forward(self, x):
        """x: [B, 3, H, W] or None → [B, out_features]

        Returns zeros if x is None or all-zero (visual not available).
        """
        if x is None:
            return torch.zeros(1, self.out_features, device=next(self.projection.parameters()).device)

        # All-zero ROI (synthetic data fallback) → skip backbone to avoid BatchNorm NaN
        if x.abs().sum() == 0:
            return torch.zeros(x.shape[0], self.out_features, device=x.device)

        if HAS_TORCHVISION:
            feats = self.backbone.features(x)
            feats = self.backbone.avgpool(feats)
            feats = torch.flatten(feats, 1)
        else:
            feats = self.backbone(x)
        return self.projection(feats)
